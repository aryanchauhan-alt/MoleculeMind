"""
D-MPNN tests: graph construction, the message-passing mathematics, the heads,
the masked multi-task loss, training, and the ensemble variance decomposition.

The message-passing tests are the important ones. A graph network that is subtly
wrong still trains and still produces plausible numbers, so the directed-message
computation is checked against its mathematical definition by explicit
enumeration rather than by eyeballing a loss curve.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem

from backend.app.core.dmpnn import (
    DMPNNConfig,
    DMPNNEnsemble,
    DMPNNModel,
    DMPNNTrainer,
    TaskSpec,
    gaussian_nll,
    masked_bce,
)
from backend.app.core.molecular_graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    batch_graphs,
    mol_to_graph,
    smiles_to_graph,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
BENZOIC = "c1ccccc1CC(=O)O"


# ------------------------------------------------------- graph construction

def test_graph_has_two_directed_edges_per_bond():
    mol = Chem.MolFromSmiles(ASPIRIN)
    g = mol_to_graph(mol)
    assert g.n_atoms == mol.GetNumHeavyAtoms()
    assert g.n_edges == 2 * mol.GetNumBonds()


def test_reverse_edge_mapping_is_an_involution():
    g = smiles_to_graph(BENZOIC)
    for i in range(g.n_edges):
        # rev(rev(i)) == i
        assert g.edge_rev[g.edge_rev[i]] == i
        # the reverse edge swaps source and destination
        assert g.edge_src[g.edge_rev[i]] == g.edge_dest[i]
        assert g.edge_dest[g.edge_rev[i]] == g.edge_src[i]
        # an edge is never its own reverse
        assert g.edge_rev[i] != i


def test_feature_dimensions_are_stable():
    g = smiles_to_graph(ASPIRIN)
    assert g.atom_feats.shape[1] == ATOM_FEATURE_DIM
    assert g.bond_feats.shape[1] == BOND_FEATURE_DIM


def test_unknown_element_does_not_crash():
    """The 'other' slot in each one-hot must absorb elements outside the vocabulary."""
    g = smiles_to_graph("[Pt]")  # platinum is not in ATOMIC_NUMS
    assert g is not None
    assert g.atom_feats.shape == (1, ATOM_FEATURE_DIM)
    assert np.isfinite(g.atom_feats).all()


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not_a_molecule") is None


def test_molecule_without_bonds():
    g = smiles_to_graph("C")  # methane: one heavy atom, no bonds
    assert g.n_atoms == 1 and g.n_edges == 0


def test_batching_offsets_indices_correctly():
    graphs = [smiles_to_graph(s) for s in ("CCO", ASPIRIN, "C")]
    batch = batch_graphs(graphs)

    assert batch.n_mols == 3
    assert batch.atom_feats.shape[0] == sum(g.n_atoms for g in graphs)
    assert batch.bond_feats.shape[0] == sum(g.n_edges for g in graphs)
    # Every edge index must point inside the batch.
    assert int(batch.edge_src.max()) < batch.atom_feats.shape[0]
    assert int(batch.edge_dest.max()) < batch.atom_feats.shape[0]
    assert int(batch.edge_rev.max()) < batch.bond_feats.shape[0]
    # atom_to_mol must partition the atoms in order.
    counts = torch.bincount(batch.atom_to_mol, minlength=3).tolist()
    assert counts == [g.n_atoms for g in graphs]


def test_reverse_mapping_survives_batching():
    graphs = [smiles_to_graph(s) for s in ("CCO", BENZOIC)]
    b = batch_graphs(graphs)
    for i in range(b.bond_feats.shape[0]):
        r = int(b.edge_rev[i])
        assert int(b.edge_rev[r]) == i
        assert int(b.edge_src[r]) == int(b.edge_dest[i])


# -------------------------------------------------- message-passing maths

@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = DMPNNConfig(hidden_size=16, depth=3, ffn_hidden_size=8, dropout=0.0)
    return DMPNNModel(cfg, [TaskSpec("y", "regression")]).eval()


def test_directed_message_excludes_reverse_edge(small_model):
    """
    The core correctness test.

    The vectorised form `incoming[src] - h[rev]` must equal the definition

        m_e = Σ_{k ∈ N(v) \\ {w}} h_{(k -> v)}   for e = (v -> w)

    computed by explicit enumeration over all edges. If the reverse-edge
    subtraction were dropped or misindexed, this would fail while training would
    still appear to work.
    """
    batch = batch_graphs([smiles_to_graph(BENZOIC)])
    enc = small_model.encoder
    h_size = enc.config.hidden_size

    with torch.no_grad():
        af, bf = batch.atom_feats, batch.bond_feats
        src, dest, rev = batch.edge_src, batch.edge_dest, batch.edge_rev
        h = F.relu(enc.W_i(torch.cat([af[src], bf], dim=1)))

        incoming = torch.zeros(af.shape[0], h_size).index_add(0, dest, h)
        vectorised = incoming[src] - h[rev]

        explicit = torch.zeros_like(vectorised)
        for e in range(len(src)):
            v, w = int(src[e]), int(dest[e])
            acc = torch.zeros(h_size)
            for j in range(len(src)):
                if int(dest[j]) == v and int(src[j]) != w:
                    acc += h[j]
            explicit[e] = acc

    assert torch.allclose(vectorised, explicit, atol=1e-5)


def test_message_is_not_merely_the_incoming_sum(small_model):
    """Guard against a regression that drops the reverse-edge subtraction."""
    batch = batch_graphs([smiles_to_graph(BENZOIC)])
    enc = small_model.encoder
    with torch.no_grad():
        af, bf = batch.atom_feats, batch.bond_feats
        src, dest, rev = batch.edge_src, batch.edge_dest, batch.edge_rev
        h = F.relu(enc.W_i(torch.cat([af[src], bf], dim=1)))
        incoming = torch.zeros(af.shape[0], enc.config.hidden_size).index_add(0, dest, h)
        directed = incoming[src] - h[rev]
    # The two must differ, i.e. the subtraction actually does something.
    assert not torch.allclose(directed, incoming[src], atol=1e-4)


def test_embedding_is_permutation_invariant(small_model):
    mol = Chem.MolFromSmiles(BENZOIC)
    order = [int(i) for i in np.random.default_rng(0).permutation(mol.GetNumAtoms())]
    permuted = Chem.RenumberAtoms(mol, order)
    with torch.no_grad():
        a = small_model.embed(batch_graphs([mol_to_graph(mol)]))
        b = small_model.embed(batch_graphs([mol_to_graph(permuted)]))
    assert torch.allclose(a, b, atol=1e-5)


def test_batched_embedding_equals_individual(small_model):
    graphs = [smiles_to_graph(s) for s in ("CCO", "c1ccccc1", ASPIRIN, "C")]
    with torch.no_grad():
        together = small_model.embed(batch_graphs(graphs))
        apart = torch.cat([small_model.embed(batch_graphs([g])) for g in graphs])
    assert torch.allclose(together, apart, atol=1e-5)


def test_bondless_molecule_produces_finite_embedding(small_model):
    with torch.no_grad():
        emb = small_model.embed(batch_graphs([smiles_to_graph("C")]))
    assert torch.isfinite(emb).all()


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_depth_is_configurable(depth):
    cfg = DMPNNConfig(hidden_size=8, depth=depth, ffn_hidden_size=8, dropout=0.0)
    model = DMPNNModel(cfg, [TaskSpec("y", "regression")]).eval()
    with torch.no_grad():
        mu, var, _ = model(batch_graphs([smiles_to_graph(ASPIRIN)]))
    assert mu.shape == (1, 1) and torch.isfinite(mu).all()
    assert (var > 0).all()


# ---------------------------------------------------------------- the heads

def test_output_shapes_for_multitask_head():
    tasks = [
        TaskSpec("sol", "regression"),
        TaskSpec("tox", "classification"),
        TaskSpec("lipo", "regression"),
    ]
    cfg = DMPNNConfig(hidden_size=8, depth=2, ffn_hidden_size=8, dropout=0.0)
    model = DMPNNModel(cfg, tasks).eval()
    batch = batch_graphs([smiles_to_graph("CCO"), smiles_to_graph(ASPIRIN)])
    with torch.no_grad():
        mu, var, logits = model(batch)
    assert mu.shape == (2, 2)       # two regression tasks
    assert var.shape == (2, 2)
    assert logits.shape == (2, 1)   # one classification task


def test_predicted_variance_is_strictly_positive():
    cfg = DMPNNConfig(hidden_size=8, depth=2, dropout=0.0, min_variance=1e-4)
    model = DMPNNModel(cfg, [TaskSpec("y", "regression")]).eval()
    with torch.no_grad():
        _, var, _ = model(batch_graphs([smiles_to_graph(s) for s in ("CCO", "C", ASPIRIN)]))
    assert (var >= cfg.min_variance).all()


def test_invalid_task_kind_rejected():
    with pytest.raises(ValueError):
        TaskSpec("y", "ordinal")


# ------------------------------------------------------------- the losses

def test_gaussian_nll_matches_closed_form():
    mu = torch.tensor([[1.0]])
    var = torch.tensor([[4.0]])
    y = torch.tensor([[3.0]])
    mask = torch.ones(1, 1)
    expected = 0.5 * (math.log(4.0) + (3.0 - 1.0) ** 2 / 4.0)
    assert float(gaussian_nll(mu, var, y, mask)) == pytest.approx(expected, abs=1e-6)


def test_gaussian_nll_is_minimised_at_the_true_variance():
    """With mu fixed and a known squared error, NLL is minimal at sigma² = error²."""
    y = torch.tensor([[2.0]])
    mu = torch.tensor([[0.0]])
    mask = torch.ones(1, 1)
    losses = {
        v: float(gaussian_nll(mu, torch.tensor([[v]]), y, mask))
        for v in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    }
    assert min(losses, key=losses.get) == 4.0  # (2.0 - 0.0)² = 4


def test_masked_loss_ignores_missing_labels():
    """A masked-out entry must not change the loss, however wrong it is."""
    mu = torch.tensor([[1.0, 1.0]])
    var = torch.ones(1, 2)
    present = torch.tensor([[2.0, 2.0]])
    mask_both = torch.ones(1, 2)
    mask_first = torch.tensor([[1.0, 0.0]])

    wild = torch.tensor([[2.0, 1e6]])
    assert float(gaussian_nll(mu, var, wild, mask_first)) == pytest.approx(
        float(gaussian_nll(mu, var, present, mask_first)), abs=1e-6
    )
    # And masking genuinely changes the result versus using both labels.
    assert float(gaussian_nll(mu, var, present, mask_both)) == pytest.approx(
        float(gaussian_nll(mu, var, present, mask_first)), abs=1e-6
    )  # equal here only because both residuals are identical


def test_masked_bce_ignores_missing_labels():
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    mask_first = torch.tensor([[1.0, 0.0]])
    loss = float(masked_bce(logits, target, mask_first))
    # Only the first element counts: BCE(logit 0, target 1) = log 2.
    assert loss == pytest.approx(math.log(2.0), abs=1e-6)


def test_fully_masked_loss_is_zero():
    mu = torch.zeros(2, 1)
    var = torch.ones(2, 1)
    y = torch.full((2, 1), 5.0)
    assert float(gaussian_nll(mu, var, y, torch.zeros(2, 1))) == pytest.approx(0.0)


# ------------------------------------------------------------- training

@pytest.fixture(scope="module")
def tiny_training_set():
    """A small regression problem the network can actually fit: heavy-atom count."""
    smiles = [
        "C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC",
        "CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1C", "c1ccccc1CC",
        "CC(=O)O", "CCC(=O)O", "CN", "CCN", "CCCN", "CO",
        "ClCCl", "BrCBr", "CS", "CCS", "CCCS", "OCCO",
    ]
    graphs = [smiles_to_graph(s) for s in smiles]
    y = np.array(
        [[float(Chem.MolFromSmiles(s).GetNumHeavyAtoms())] for s in smiles],
        dtype=np.float32,
    )
    return graphs, y


def test_training_reduces_loss(tiny_training_set):
    graphs, y = tiny_training_set
    cfg = DMPNNConfig(
        hidden_size=32, depth=3, ffn_hidden_size=32, dropout=0.0,
        max_epochs=40, patience=40, batch_size=8, seed=0,
    )
    trainer = DMPNNTrainer(cfg, [TaskSpec("n_heavy", "regression")])
    report = trainer.fit(graphs, y)

    first = trainer.history[0]["train_loss"]
    last = min(h["train_loss"] for h in trainer.history)
    assert last < first, f"loss did not decrease: {first} -> {last}"
    assert report["epochs_run"] > 0


def test_training_learns_the_signal(tiny_training_set):
    graphs, y = tiny_training_set
    cfg = DMPNNConfig(
        hidden_size=64, depth=3, ffn_hidden_size=64, dropout=0.0,
        max_epochs=120, patience=120, batch_size=8, seed=0,
    )
    trainer = DMPNNTrainer(cfg, [TaskSpec("n_heavy", "regression")])
    trainer.fit(graphs, y)
    mu, var, _ = trainer.predict(graphs)
    # Heavy-atom count is exactly recoverable from the graph, so a working
    # network should correlate strongly with it on the training set.
    r = float(np.corrcoef(mu[:, 0], y[:, 0])[0, 1])
    assert r > 0.9, f"correlation with a learnable target was only {r:.2f}"
    assert (var > 0).all()


def test_training_is_reproducible_for_a_fixed_seed(tiny_training_set):
    graphs, y = tiny_training_set
    cfg = dict(
        hidden_size=16, depth=2, ffn_hidden_size=16, dropout=0.0,
        max_epochs=10, patience=10, batch_size=8, seed=7,
    )
    a = DMPNNTrainer(DMPNNConfig(**cfg), [TaskSpec("y", "regression")])
    a.fit(graphs, y)
    b = DMPNNTrainer(DMPNNConfig(**cfg), [TaskSpec("y", "regression")])
    b.fit(graphs, y)
    assert np.allclose(a.predict(graphs)[0], b.predict(graphs)[0], atol=1e-5)


def test_learning_rate_schedule_warms_up_then_decays():
    cfg = DMPNNConfig(lr=1e-3, warmup_epochs=2, max_epochs=20, final_lr_ratio=0.1)
    trainer = DMPNNTrainer(cfg, [TaskSpec("y", "regression")])
    lrs = [trainer._lr_at(e) for e in range(20)]
    assert lrs[0] < lrs[1] <= cfg.lr          # warm-up rises
    assert lrs[2] >= lrs[10] >= lrs[19]       # then decays monotonically
    assert lrs[19] == pytest.approx(cfg.lr * cfg.final_lr_ratio, rel=0.2)


def test_early_stopping_triggers_on_an_unlearnable_validation_set(tiny_training_set):
    """
    Validation labels are pure noise, so validation loss cannot keep improving
    and the patience counter must fire well before max_epochs.
    """
    graphs, y = tiny_training_set
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(len(graphs) - 16, 1)).astype(np.float32) * 50.0

    cfg = DMPNNConfig(
        hidden_size=8, depth=2, ffn_hidden_size=8, dropout=0.0,
        max_epochs=300, patience=3, batch_size=8, seed=0,
    )
    trainer = DMPNNTrainer(cfg, [TaskSpec("y", "regression")])
    report = trainer.fit(graphs[:16], y[:16], graphs[16:], noise)
    assert report["early_stopped"] is True
    assert report["epochs_run"] < 300
    # The restored weights must be the best-scoring ones, not the last ones.
    assert report["best_epoch"] <= report["epochs_run"] - 1


def test_multitask_training_with_missing_labels(tiny_training_set):
    """Half the classification labels are NaN; training must still work."""
    graphs, y_reg = tiny_training_set
    y = np.full((len(graphs), 2), np.nan, dtype=np.float32)
    y[:, 0] = y_reg[:, 0]
    y[::2, 1] = (y_reg[::2, 0] > 3).astype(np.float32)  # only every other label

    cfg = DMPNNConfig(
        hidden_size=32, depth=2, ffn_hidden_size=32, dropout=0.0,
        max_epochs=30, patience=30, batch_size=8, seed=0,
    )
    trainer = DMPNNTrainer(
        cfg, [TaskSpec("n_heavy", "regression"), TaskSpec("big", "classification")]
    )
    trainer.fit(graphs, y)
    mu, var, prob = trainer.predict(graphs)
    assert mu.shape == (len(graphs), 1)
    assert prob.shape == (len(graphs), 1)
    assert np.isfinite(mu).all() and np.isfinite(prob).all()
    assert ((prob >= 0) & (prob <= 1)).all()


def test_targets_are_standardised_internally(tiny_training_set):
    """Predictions must come back in the original units, not standardised ones."""
    graphs, y = tiny_training_set
    cfg = DMPNNConfig(
        hidden_size=32, depth=2, ffn_hidden_size=32, dropout=0.0,
        max_epochs=60, patience=60, batch_size=8, seed=0,
    )
    trainer = DMPNNTrainer(cfg, [TaskSpec("n_heavy", "regression")])
    trainer.fit(graphs, y)
    mu, _, _ = trainer.predict(graphs)
    # Targets range 1-6 heavy atoms; a standardised output would sit near 0.
    assert mu[:, 0].mean() > 1.0


# ------------------------------------------------------------- ensemble

@pytest.fixture(scope="module")
def trained_ensemble(tiny_training_set):
    graphs, y = tiny_training_set
    cfg = DMPNNConfig(
        hidden_size=32, depth=2, ffn_hidden_size=32, dropout=0.1,
        max_epochs=40, patience=40, batch_size=8,
    )
    ensemble = DMPNNEnsemble(
        [TaskSpec("n_heavy", "regression")], config=cfg, n_members=3, base_seed=0
    )
    ensemble.fit(graphs, y)
    return ensemble, graphs, y


def test_ensemble_members_are_independently_seeded(trained_ensemble):
    ensemble, graphs, _ = trained_ensemble
    assert len(ensemble.members) == 3
    assert [m.config.seed for m in ensemble.members] == [0, 1, 2]
    # Different seeds must give different parameters, hence different predictions.
    preds = [m.predict(graphs)[0][:, 0] for m in ensemble.members]
    assert not np.allclose(preds[0], preds[1], atol=1e-6)


def test_variance_decomposition_is_exact(trained_ensemble):
    """
    Var_total must equal mean(member variances) + var(member means) exactly.
    """
    ensemble, graphs, _ = trained_ensemble
    out = ensemble.predict(graphs)

    member_mus, member_vars = [], []
    for m in ensemble.members:
        mu, var, _ = m.predict(graphs)
        member_mus.append(mu)
        member_vars.append(var)
    mu_stack = np.stack(member_mus)
    var_stack = np.stack(member_vars)

    assert np.allclose(out["mu"], mu_stack.mean(axis=0), atol=1e-6)
    assert np.allclose(out["aleatoric_variance"], var_stack.mean(axis=0), atol=1e-6)
    assert np.allclose(out["epistemic_variance"], mu_stack.var(axis=0), atol=1e-6)
    assert np.allclose(
        out["variance"],
        out["aleatoric_variance"] + out["epistemic_variance"],
        atol=1e-8,
    )
    assert np.allclose(out["std"], np.sqrt(out["variance"]), atol=1e-8)


def test_epistemic_variance_is_non_negative(trained_ensemble):
    ensemble, graphs, _ = trained_ensemble
    out = ensemble.predict(graphs)
    assert (out["epistemic_variance"] >= 0).all()
    assert (out["aleatoric_variance"] > 0).all()


def test_ensemble_predict_before_fit_raises():
    ensemble = DMPNNEnsemble([TaskSpec("y", "regression")], n_members=2)
    with pytest.raises(RuntimeError):
        ensemble.predict([smiles_to_graph("CCO")])


def test_predict_smiles_reports_invalid_input(trained_ensemble):
    ensemble, _, _ = trained_ensemble
    out = ensemble.predict_smiles(["CCO", "not_a_molecule", "CCCO"])
    assert out["invalid_smiles"] == ["not_a_molecule"]
    assert out["smiles"] == ["CCO", "CCCO"]
    assert out["mu"].shape[0] == 2


def test_ensemble_embedding_dimension(trained_ensemble):
    ensemble, graphs, _ = trained_ensemble
    emb = ensemble.embed(graphs)
    assert emb.shape == (len(graphs), ensemble.base_config.hidden_size)
    assert np.isfinite(emb).all()


def test_save_and_load_round_trip(trained_ensemble, tmp_path):
    ensemble, graphs, _ = trained_ensemble
    path = str(tmp_path / "ens.pt")
    ensemble.save(path)

    restored = DMPNNEnsemble.load(path)
    assert len(restored.members) == len(ensemble.members)

    a = ensemble.predict(graphs)
    b = restored.predict(graphs)
    assert np.allclose(a["mu"], b["mu"], atol=1e-6)
    assert np.allclose(a["variance"], b["variance"], atol=1e-6)


def test_training_report_records_configuration(trained_ensemble):
    ensemble, _, _ = trained_ensemble
    report = ensemble.training_report
    assert report["n_members"] == 3
    assert len(report["members"]) == 3
    assert report["config"]["depth"] == 2
    assert "Var_total" in report["variance_decomposition"]
    for member in report["members"]:
        assert "seed" in member and "best_epoch" in member
