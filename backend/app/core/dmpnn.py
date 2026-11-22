"""
Directed Message Passing Neural Network (D-MPNN) with an uncertainty-aware
multi-task head, plus a deep-ensemble wrapper.

Architecture (per the project methodology)
------------------------------------------
* 3 message-passing steps (configurable 3-5)
* hidden size 300
* mean readout over atoms
* two-layer feed-forward head, ReLU, dropout 0.1
* multi-task output with a masked loss for missing labels
* Gaussian negative log-likelihood for regression, BCE-with-logits for classification
* Adam, learning-rate warm-up then decay from 1e-3, batch size 64, early stopping
* five independently seeded ensemble members

Mathematics of the message passing
----------------------------------
Let `e = (v -> w)` be a directed edge, `x_v` the atom features of `v` and
`f_vw` the bond features.

1. **Edge initialisation**

       h_e^0 = ReLU( W_i · [x_v ; f_vw] )

2. **Message for edge `e = (v -> w)`** — the sum of messages arriving at `v`,
   *excluding* the reverse edge `w -> v`:

       m_e^t = Σ_{k ∈ N(v) \\ {w}} h_{(k -> v)}^t
             = ( Σ_{k ∈ N(v)} h_{(k -> v)}^t ) − h_{(w -> v)}^t

   The subtraction is the whole point of the *directed* formulation. Without
   it, a message would flow straight back along the bond it arrived on and the
   network would amplify 2-step walks instead of exploring the graph. This is
   implemented as `incoming[edge_src] − h[edge_rev]`, which is exactly the
   closed form above and is what `test_directed_message_excludes_reverse_edge`
   verifies numerically.

3. **Edge update**, with a residual onto the initial embedding (this is what
   keeps deep stacks stable and is the form used in the reference
   implementation):

       h_e^{t+1} = ReLU( h_e^0 + W_h · m_e^t )

4. **Atom readout** — gather all messages arriving at each atom:

       m_v = Σ_{k ∈ N(v)} h_{(k -> v)}^T
       h_v = ReLU( W_o · [x_v ; m_v] )

5. **Molecule embedding** — mean over the molecule's atoms.

Uncertainty
-----------
Each regression task emits a mean `mu` and a log-variance, trained by Gaussian
NLL, so the model learns *aleatoric* (data) uncertainty. Across an ensemble of
`M` independently initialised members, total predictive variance decomposes as

    Var_total = (1/M) Σ_m sigma_m²        <- mean aleatoric component
                + (1/M) Σ_m (mu_m − mu̅)²  <- epistemic component (spread of means)

which is the standard deep-ensemble decomposition (Lakshminarayanan et al. 2017)
and the combination rule the methodology specifies. `mu̅` is the ensemble mean.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.core.molecular_graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    BatchedMolGraph,
    MolGraph,
    batch_graphs,
    smiles_to_graph,
)


# --------------------------------------------------------------------- config

@dataclass
class TaskSpec:
    """One prediction task on the shared molecular representation."""

    name: str
    kind: str  # "regression" | "classification"

    def __post_init__(self):
        if self.kind not in ("regression", "classification"):
            raise ValueError(f"Unknown task kind: {self.kind}")


@dataclass
class DMPNNConfig:
    hidden_size: int = 300
    depth: int = 3                  # message-passing steps (methodology: 3-5)
    ffn_hidden_size: int = 300
    dropout: float = 0.1
    # Optimisation
    lr: float = 1e-3
    warmup_epochs: int = 2
    max_epochs: int = 60
    batch_size: int = 64
    patience: int = 10              # early-stopping patience in epochs
    final_lr_ratio: float = 0.1     # decay to lr * ratio by the last epoch
    grad_clip: float = 5.0
    # Numerical floor on predicted variance, so NLL cannot diverge.
    min_variance: float = 1e-4
    seed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- the network

class DMPNNEncoder(nn.Module):
    """Directed message passing over bonds, producing a molecule embedding."""

    def __init__(self, config: DMPNNConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.W_i = nn.Linear(ATOM_FEATURE_DIM + BOND_FEATURE_DIM, h, bias=False)
        self.W_h = nn.Linear(h, h, bias=False)
        self.W_o = nn.Linear(ATOM_FEATURE_DIM + h, h)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, batch: BatchedMolGraph) -> torch.Tensor:
        """Returns (n_mols, hidden_size) molecule embeddings."""
        atom_feats = batch.atom_feats
        n_atoms = atom_feats.shape[0]
        device = atom_feats.device
        h_size = self.config.hidden_size

        if batch.bond_feats.shape[0] == 0:
            # Edge case: no bonds anywhere in the batch (e.g. a batch of single
            # ions). Readout still works with a zero message vector.
            atom_messages = torch.zeros(n_atoms, h_size, device=device)
        else:
            src, dest, rev = batch.edge_src, batch.edge_dest, batch.edge_rev

            # 1. Edge initialisation: h_e^0 = ReLU(W_i [x_v ; f_vw])
            edge_input = torch.cat([atom_feats[src], batch.bond_feats], dim=1)
            h_0 = F.relu(self.W_i(edge_input))
            h = h_0

            # 2-3. Message passing
            for _ in range(self.config.depth - 1):
                # Sum of messages arriving at each atom.
                incoming = torch.zeros(n_atoms, h_size, device=device)
                incoming = incoming.index_add(0, dest, h)
                # Directed message: exclude the reverse edge.
                message = incoming[src] - h[rev]
                h = F.relu(h_0 + self.W_h(message))
                h = self.dropout(h)

            # 4. Atom readout: gather all messages arriving at each atom.
            atom_messages = torch.zeros(n_atoms, h_size, device=device)
            atom_messages = atom_messages.index_add(0, dest, h)

        atom_hidden = F.relu(self.W_o(torch.cat([atom_feats, atom_messages], dim=1)))
        atom_hidden = self.dropout(atom_hidden)

        # 5. Mean readout per molecule.
        mol_sum = torch.zeros(batch.n_mols, h_size, device=device)
        mol_sum = mol_sum.index_add(0, batch.atom_to_mol, atom_hidden)
        counts = torch.zeros(batch.n_mols, device=device)
        counts = counts.index_add(
            0, batch.atom_to_mol, torch.ones(n_atoms, device=device)
        )
        return mol_sum / counts.clamp(min=1.0).unsqueeze(1)


class DMPNNModel(nn.Module):
    """
    Encoder plus a two-layer multi-task head.

    Regression tasks emit (mu, log_var); classification tasks emit a logit.
    Output width is therefore `2 * n_regression + n_classification`.
    """

    def __init__(self, config: DMPNNConfig, tasks: Sequence[TaskSpec]):
        super().__init__()
        self.config = config
        self.tasks = list(tasks)
        self.reg_idx = [i for i, t in enumerate(self.tasks) if t.kind == "regression"]
        self.clf_idx = [i for i, t in enumerate(self.tasks) if t.kind == "classification"]

        out_dim = 2 * len(self.reg_idx) + len(self.clf_idx)
        self.encoder = DMPNNEncoder(config)
        self.ffn = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_hidden_size, out_dim),
        )

    def embed(self, batch: BatchedMolGraph) -> torch.Tensor:
        """Molecule embeddings — the learned latent space used by the AD module."""
        return self.encoder(batch)

    def forward(
        self, batch: BatchedMolGraph
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (mu, variance, logits):
          mu, variance : (n_mols, n_regression)
          logits       : (n_mols, n_classification)
        """
        out = self.ffn(self.encoder(batch))
        n_reg = len(self.reg_idx)

        if n_reg:
            mu = out[:, :n_reg]
            log_var = out[:, n_reg : 2 * n_reg]
            # softplus keeps variance positive; the floor keeps NLL finite.
            variance = F.softplus(log_var) + self.config.min_variance
        else:
            mu = out[:, :0]
            variance = out[:, :0]

        logits = out[:, 2 * n_reg :]
        return mu, variance, logits


# ------------------------------------------------------------------- training

def gaussian_nll(
    mu: torch.Tensor,
    variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked Gaussian negative log-likelihood, up to the additive constant:

        NLL = 0.5 * [ log(sigma²) + (y - mu)² / sigma² ]

    `mask` is 1 where a label is present. Missing labels contribute nothing, so
    a multi-task batch with sparse labels trains correctly without imputation.
    """
    if mu.numel() == 0:
        return mu.sum()
    nll = 0.5 * (torch.log(variance) + (target - mu) ** 2 / variance)
    nll = nll * mask
    denom = mask.sum().clamp(min=1.0)
    return nll.sum() / denom


def masked_bce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked binary cross-entropy with logits."""
    if logits.numel() == 0:
        return logits.sum()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


class _Dataset:
    """Graphs plus a (n, n_tasks) target matrix where NaN marks a missing label."""

    def __init__(self, graphs: List[MolGraph], targets: np.ndarray):
        if len(graphs) != targets.shape[0]:
            raise ValueError("graphs and targets must have the same length.")
        self.graphs = graphs
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.graphs)


def _iterate_batches(
    dataset: _Dataset, batch_size: int, rng: Optional[np.random.Generator] = None
):
    order = np.arange(len(dataset))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = batch_graphs([dataset.graphs[i] for i in idx])
        yield batch, torch.from_numpy(dataset.targets[idx])


class DMPNNTrainer:
    """Trains one D-MPNN member with warm-up, decay and early stopping."""

    def __init__(self, config: DMPNNConfig, tasks: Sequence[TaskSpec]):
        self.config = config
        self.tasks = list(tasks)
        # The seed must be set BEFORE the model is constructed: weight
        # initialisation draws from the global RNG, so seeding only inside
        # fit() left initialisation at the mercy of ambient state and made
        # runs with an identical seed produce different models.
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.model = DMPNNModel(config, tasks)
        self.history: List[Dict[str, float]] = []
        # Targets are standardised per regression task; predictions are mapped
        # back afterwards. Without this, a task measured in log-units and one in
        # kcal/mol would contribute wildly unequal gradients.
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None

    # -- learning-rate schedule --------------------------------------------
    def _lr_at(self, epoch: int) -> float:
        cfg = self.config
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            # Linear warm-up from lr/10 to lr.
            frac = (epoch + 1) / cfg.warmup_epochs
            return cfg.lr * (0.1 + 0.9 * frac)
        # Exponential decay from lr to lr * final_lr_ratio over the remainder.
        total = max(cfg.max_epochs - cfg.warmup_epochs, 1)
        progress = min((epoch - cfg.warmup_epochs) / total, 1.0)
        return cfg.lr * (cfg.final_lr_ratio ** progress)

    def _standardise(self, targets: np.ndarray) -> np.ndarray:
        reg = self.model.reg_idx
        out = targets.copy()
        if not reg:
            return out
        self.target_mean = np.zeros(targets.shape[1], dtype=np.float32)
        self.target_std = np.ones(targets.shape[1], dtype=np.float32)
        for i in reg:
            col = targets[:, i]
            valid = col[~np.isnan(col)]
            if valid.size:
                self.target_mean[i] = float(valid.mean())
                self.target_std[i] = float(valid.std()) or 1.0
            out[:, i] = (col - self.target_mean[i]) / self.target_std[i]
        return out

    def _loss(self, batch: BatchedMolGraph, targets: torch.Tensor) -> torch.Tensor:
        mu, variance, logits = self.model(batch)
        total = torch.zeros((), dtype=torch.float32)

        if self.model.reg_idx:
            cols = targets[:, self.model.reg_idx]
            mask = (~torch.isnan(cols)).float()
            total = total + gaussian_nll(
                mu, variance, torch.nan_to_num(cols, nan=0.0), mask
            )
        if self.model.clf_idx:
            cols = targets[:, self.model.clf_idx]
            mask = (~torch.isnan(cols)).float()
            total = total + masked_bce(
                logits, torch.nan_to_num(cols, nan=0.0), mask
            )
        return total

    def fit(
        self,
        train_graphs: List[MolGraph],
        train_targets: np.ndarray,
        val_graphs: Optional[List[MolGraph]] = None,
        val_targets: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        train_std_targets = self._standardise(train_targets)
        train_ds = _Dataset(train_graphs, train_std_targets)

        val_ds = None
        if val_graphs is not None and val_targets is not None and len(val_graphs):
            val_std = val_targets.copy()
            for i in self.model.reg_idx:
                val_std[:, i] = (val_targets[:, i] - self.target_mean[i]) / self.target_std[i]
            val_ds = _Dataset(val_graphs, val_std)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        best_loss = math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = -1
        epochs_without_improvement = 0
        started = time.time()

        for epoch in range(cfg.max_epochs):
            lr = self._lr_at(epoch)
            for group in optimiser.param_groups:
                group["lr"] = lr

            self.model.train()
            running, n_batches = 0.0, 0
            for batch, targets in _iterate_batches(train_ds, cfg.batch_size, rng):
                optimiser.zero_grad()
                loss = self._loss(batch, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                optimiser.step()
                running += float(loss.item())
                n_batches += 1
            train_loss = running / max(n_batches, 1)

            if val_ds is not None:
                val_loss = self._evaluate_loss(val_ds)
                monitored = val_loss
            else:
                val_loss = float("nan")
                monitored = train_loss

            self.history.append(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 5) if val_ds is not None else None,
                }
            )
            if verbose:
                print(
                    f"  epoch {epoch:3d}  lr {lr:.2e}  train {train_loss:.4f}"
                    + (f"  val {val_loss:.4f}" if val_ds is not None else "")
                )

            if monitored < best_loss - 1e-5:
                best_loss = monitored
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "epochs_run": len(self.history),
            "best_epoch": best_epoch,
            "best_loss": round(best_loss, 5),
            "early_stopped": epochs_without_improvement >= cfg.patience,
            "seconds": round(time.time() - started, 1),
            "history": self.history,
        }

    def _evaluate_loss(self, dataset: _Dataset) -> float:
        self.model.eval()
        running, n_batches = 0.0, 0
        with torch.no_grad():
            for batch, targets in _iterate_batches(dataset, self.config.batch_size):
                running += float(self._loss(batch, targets).item())
                n_batches += 1
        return running / max(n_batches, 1)

    # -- inference ---------------------------------------------------------
    def predict(
        self, graphs: List[MolGraph], batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mu, variance, probabilities) in the ORIGINAL target units.

        Variance is rescaled by std² because Var(a·X) = a²·Var(X).
        """
        self.model.eval()
        bs = batch_size or self.config.batch_size
        mus, variances, probs = [], [], []

        with torch.no_grad():
            for start in range(0, len(graphs), bs):
                chunk = graphs[start : start + bs]
                batch = batch_graphs(chunk)
                mu, var, logits = self.model(batch)
                mus.append(mu.numpy())
                variances.append(var.numpy())
                probs.append(torch.sigmoid(logits).numpy())

        mu = np.concatenate(mus, axis=0) if mus else np.zeros((0, 0))
        var = np.concatenate(variances, axis=0) if variances else np.zeros((0, 0))
        prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, 0))

        for k, i in enumerate(self.model.reg_idx):
            mu[:, k] = mu[:, k] * self.target_std[i] + self.target_mean[i]
            var[:, k] = var[:, k] * (self.target_std[i] ** 2)
        return mu, var, prob

    def embed(self, graphs: List[MolGraph], batch_size: Optional[int] = None) -> np.ndarray:
        """Learned latent representation, for the latent-space applicability domain."""
        self.model.eval()
        bs = batch_size or self.config.batch_size
        out = []
        with torch.no_grad():
            for start in range(0, len(graphs), bs):
                out.append(self.model.embed(batch_graphs(graphs[start : start + bs])).numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.config.hidden_size))


# ------------------------------------------------------------------ ensemble

class DMPNNEnsemble:
    """
    Deep ensemble of independently seeded D-MPNN members.

    Total predictive variance combines both uncertainty sources:

        Var_total = mean(sigma_m²) + var(mu_m)

    the first term aleatoric (each member's own learned noise), the second
    epistemic (disagreement between members).
    """

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        config: Optional[DMPNNConfig] = None,
        n_members: int = 5,
        base_seed: int = 42,
    ):
        self.tasks = list(tasks)
        self.base_config = config or DMPNNConfig()
        self.n_members = n_members
        self.base_seed = base_seed
        self.members: List[DMPNNTrainer] = []
        self.training_report: Dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return len(self.members) > 0

    def fit(
        self,
        train_graphs: List[MolGraph],
        train_targets: np.ndarray,
        val_graphs: Optional[List[MolGraph]] = None,
        val_targets: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        self.members = []
        member_reports = []
        started = time.time()

        for m in range(self.n_members):
            cfg = DMPNNConfig(**{**self.base_config.as_dict(), "seed": self.base_seed + m})
            if verbose:
                print(f"[D-MPNN] training member {m + 1}/{self.n_members} (seed {cfg.seed})")
            trainer = DMPNNTrainer(cfg, self.tasks)
            report = trainer.fit(
                train_graphs, train_targets, val_graphs, val_targets, verbose=verbose
            )
            report["seed"] = cfg.seed
            report.pop("history", None)  # keep the summary compact
            member_reports.append(report)
            self.members.append(trainer)

        self.training_report = {
            "n_members": self.n_members,
            "base_seed": self.base_seed,
            "config": self.base_config.as_dict(),
            "tasks": [asdict(t) for t in self.tasks],
            "n_train": len(train_graphs),
            "n_val": len(val_graphs) if val_graphs else 0,
            "total_seconds": round(time.time() - started, 1),
            "members": member_reports,
            "variance_decomposition": (
                "Var_total = mean_m(sigma_m^2) + var_m(mu_m): the first term is the "
                "mean aleatoric variance each member predicts, the second is the "
                "epistemic spread between member means."
            ),
        }
        return self.training_report

    def predict(self, graphs: List[MolGraph]) -> Dict[str, np.ndarray]:
        """
        Ensemble prediction with a decomposed variance.

        Returns a dict with:
          mu                  (n, n_reg)  ensemble mean
          variance            (n, n_reg)  total predictive variance
          aleatoric_variance  (n, n_reg)  mean of member variances
          epistemic_variance  (n, n_reg)  variance of member means
          std                 (n, n_reg)  sqrt(total)
          probability         (n, n_clf)  mean predicted probability
          probability_std     (n, n_clf)  spread across members
        """
        if not self.is_trained:
            raise RuntimeError("DMPNNEnsemble.predict() called before fit().")

        mus, variances, probs = [], [], []
        for trainer in self.members:
            mu, var, prob = trainer.predict(graphs)
            mus.append(mu)
            variances.append(var)
            probs.append(prob)

        mu_stack = np.stack(mus)            # (M, n, n_reg)
        var_stack = np.stack(variances)
        prob_stack = np.stack(probs)

        mu_mean = mu_stack.mean(axis=0)
        aleatoric = var_stack.mean(axis=0)
        epistemic = mu_stack.var(axis=0)
        total = aleatoric + epistemic

        return {
            "mu": mu_mean,
            "variance": total,
            "aleatoric_variance": aleatoric,
            "epistemic_variance": epistemic,
            "std": np.sqrt(total),
            "probability": prob_stack.mean(axis=0),
            "probability_std": prob_stack.std(axis=0),
            "member_means": mu_stack,
        }

    def predict_smiles(self, smiles_list: Sequence[str]) -> Dict[str, Any]:
        """Convenience wrapper; unparseable SMILES are reported, not silently dropped."""
        graphs, kept, invalid = [], [], []
        for s in smiles_list:
            g = smiles_to_graph(s)
            if g is None:
                invalid.append(s)
            else:
                graphs.append(g)
                kept.append(s)
        if not graphs:
            return {"smiles": [], "invalid_smiles": invalid, "predictions": None}
        out = self.predict(graphs)
        out["smiles"] = kept
        out["invalid_smiles"] = invalid
        return out

    def embed(self, graphs: List[MolGraph]) -> np.ndarray:
        """Mean latent embedding across members."""
        if not self.is_trained:
            raise RuntimeError("DMPNNEnsemble.embed() called before fit().")
        return np.stack([m.embed(graphs) for m in self.members]).mean(axis=0)

    # -- persistence -------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "tasks": [asdict(t) for t in self.tasks],
                "base_config": self.base_config.as_dict(),
                "n_members": self.n_members,
                "base_seed": self.base_seed,
                "training_report": self.training_report,
                "members": [
                    {
                        "state_dict": m.model.state_dict(),
                        "config": m.config.as_dict(),
                        "target_mean": m.target_mean,
                        "target_std": m.target_std,
                    }
                    for m in self.members
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DMPNNEnsemble":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        tasks = [TaskSpec(**t) for t in blob["tasks"]]
        ensemble = cls(
            tasks=tasks,
            config=DMPNNConfig(**blob["base_config"]),
            n_members=blob["n_members"],
            base_seed=blob["base_seed"],
        )
        ensemble.training_report = blob.get("training_report", {})
        for member_blob in blob["members"]:
            cfg = DMPNNConfig(**member_blob["config"])
            trainer = DMPNNTrainer(cfg, tasks)
            trainer.model.load_state_dict(member_blob["state_dict"])
            trainer.target_mean = member_blob["target_mean"]
            trainer.target_std = member_blob["target_std"]
            ensemble.members.append(trainer)
        return ensemble
