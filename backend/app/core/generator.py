"""
SMILES molecular generator with REINVENT-style reinforcement learning.

Honest statement of scope
-------------------------
The methodology specifies pre-training on MOSES (~1.9M molecules). **MOSES is not
bundled with this project and is not downloaded automatically.** The prior here
is trained on the SMILES available locally - the union of Delaney ESOL and
ClinTox, about 2,500 molecules - which is three orders of magnitude smaller.

The consequence is concrete and must not be glossed over: a character-level
language model trained on 2,500 SMILES learns the grammar only partially, so
**validity rates will be far below the >90% a MOSES-pretrained prior reaches.**
Every number this module reports is measured on the actual trained model, so the
gap is visible rather than hidden. The architecture, the RL objective, the
diversity filter and the evaluation metrics are all fully implemented; what is
missing is data, and `prior_data_limitation()` says so in the API payload.

`scripts/download_moses.py` documents how to obtain MOSES and point the trainer
at it, at which point the same code trains a competitive prior.

Architecture
------------
Character-level tokenisation with multi-character atom tokens handled (Cl, Br,
bracket atoms), a GRU language model, teacher-forced maximum-likelihood
pre-training, then REINVENT augmented-likelihood RL.

REINVENT augmented likelihood (Olivecrona et al. 2017)
------------------------------------------------------
The agent is initialised from the prior and optimised towards

    log P_augmented(S) = log P_prior(S) + sigma * R(S)

    Loss = [ log P_augmented(S) - log P_agent(S) ]^2

The prior term is what prevents reward hacking. A bare policy-gradient agent
maximising R alone drifts into degenerate strings that score well and are not
molecules; anchoring to the prior's likelihood keeps generated SMILES in
chemically plausible territory, and `sigma` sets how far the agent may stray.

Uncertainty-aware reward
------------------------
    R = w_valid   * validity
      + w_obj     * mean over objectives of a LOWER-CONFIDENCE-BOUND score
      + w_ad      * applicability-domain term
      - w_dup     * duplicate penalty

The objective term deliberately uses the **lower** conformal bound, not the
mean: a candidate is rewarded for what the model is *confident* about, not for
an optimistic guess. Rewarding the mean teaches the generator to exploit regions
where the model is uncertain and happens to predict well, which is precisely the
failure mode uncertainty-aware generation is meant to avoid.

The diversity filter penalises repeated Murcko scaffolds, so the agent cannot
collect reward by emitting one good molecule over and over.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger("moleculemind.generator")

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"

# Multi-character tokens must be matched before single characters, or "Cl"
# becomes "C" + "l" and the grammar is corrupted.
_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|%\d{2}|[A-Za-z]|\d|\(|\)|\[|\]|=|#|-|\+|\\|/|:|~|\.|\*)"
)


def tokenize_smiles(smiles: str) -> List[str]:
    """Split a SMILES string into chemically meaningful tokens."""
    tokens = _TOKEN_PATTERN.findall(smiles)
    # Round-trip check: if the tokens do not reassemble the input, the string
    # contained something the pattern does not cover.
    if "".join(tokens) != smiles:
        return []
    return tokens


class SmilesVocabulary:
    def __init__(self, tokens: Optional[Sequence[str]] = None):
        specials = [PAD, BOS, EOS, UNK]
        self.itos: List[str] = list(specials) + sorted(set(tokens or []) - set(specials))
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def from_smiles(cls, smiles_list: Sequence[str]) -> "SmilesVocabulary":
        seen: set = set()
        for s in smiles_list:
            seen.update(tokenize_smiles(s))
        return cls(sorted(seen))

    def encode(self, smiles: str, add_special: bool = True) -> List[int]:
        tokens = tokenize_smiles(smiles)
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in tokens]
        if add_special:
            return [self.stoi[BOS]] + ids + [self.stoi[EOS]]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            token = self.itos[int(i)]
            if token == EOS:
                break
            if token in (BOS, PAD):
                continue
            out.append(token)
        return "".join(out)

    def as_dict(self) -> Dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "SmilesVocabulary":
        vocab = cls()
        vocab.itos = list(blob["itos"])
        vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
        return vocab


@dataclass
class GeneratorConfig:
    embedding_size: int = 128
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    max_length: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6
    seed: int = 42
    # RL
    rl_sigma: float = 60.0      # REINVENT augmented-likelihood weight
    rl_lr: float = 1e-4
    rl_batch_size: int = 32
    rl_steps: int = 50

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SmilesRNN(nn.Module):
    """GRU character-level language model over SMILES tokens."""

    def __init__(self, vocab_size: int, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.embedding_size, padding_idx=0)
        self.rnn = nn.GRU(
            config.embedding_size,
            config.hidden_size,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.out = nn.Linear(config.hidden_size, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        return self.out(out), hidden


class MolecularGenerator:
    """Prior training, sampling, evaluation and REINVENT-style RL."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.vocab: Optional[SmilesVocabulary] = None
        self.model: Optional[SmilesRNN] = None
        self.training_smiles: set = set()
        self.training_scaffolds: set = set()
        self.prior_report: Dict[str, Any] = {}
        self.rl_report: Dict[str, Any] = {}
        self.status: str = "not_trained"
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    # ------------------------------------------------------- prior training

    def train_prior(
        self,
        smiles_list: Sequence[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Teacher-forced maximum-likelihood training of the prior."""
        started = time.time()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        usable = [s for s in smiles_list if tokenize_smiles(s)]
        dropped = len(smiles_list) - len(usable)
        if len(usable) < 20:
            raise ValueError(
                f"Only {len(usable)} tokenisable SMILES available; cannot train a prior."
            )

        self.vocab = SmilesVocabulary.from_smiles(usable)
        self.training_smiles = set(usable)
        self.training_scaffolds = {
            self._scaffold(s) for s in usable if self._scaffold(s)
        }
        self.model = SmilesRNN(len(self.vocab), self.config)

        encoded = [self.vocab.encode(s) for s in usable]
        encoded = [e for e in encoded if len(e) <= self.config.max_length]

        rng = np.random.default_rng(self.config.seed)
        order = rng.permutation(len(encoded))
        split = max(int(0.9 * len(order)), 1)
        train_idx, val_idx = order[:split], order[split:]

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        best_val, best_state, bad_epochs = math.inf, None, 0
        history: List[Dict[str, float]] = []

        for epoch in range(self.config.max_epochs):
            self.model.train()
            train_loss = self._run_epoch(
                [encoded[i] for i in train_idx], optimiser, rng, train=True
            )
            self.model.eval()
            val_loss = (
                self._run_epoch([encoded[i] for i in val_idx], None, rng, train=False)
                if len(val_idx)
                else train_loss
            )
            history.append(
                {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "val_loss": round(val_loss, 4)}
            )
            if verbose:
                print(f"  epoch {epoch:3d} train {train_loss:.4f} val {val_loss:.4f}")

            if val_loss < best_val - 1e-4:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.status = "prior_trained"
        self.prior_report = {
            "status": "trained",
            "n_training_smiles": len(usable),
            "n_dropped_untokenisable": dropped,
            "vocabulary_size": len(self.vocab),
            "n_training_scaffolds": len(self.training_scaffolds),
            "epochs_run": len(history),
            "best_validation_loss": round(best_val, 4),
            "seconds": round(time.time() - started, 1),
            "config": self.config.as_dict(),
            "history": history[-10:],
            "data_limitation": self.prior_data_limitation(len(usable)),
        }
        logger.info(
            "Generator prior trained on %d SMILES (vocab %d) in %.1fs, val loss %.4f",
            len(usable), len(self.vocab), time.time() - started, best_val,
        )
        return self.prior_report

    def _run_epoch(
        self,
        sequences: List[List[int]],
        optimiser: Optional[torch.optim.Optimizer],
        rng: np.random.Generator,
        train: bool,
    ) -> float:
        if not sequences:
            return 0.0
        order = rng.permutation(len(sequences)) if train else np.arange(len(sequences))
        total, n_batches = 0.0, 0

        for start in range(0, len(order), self.config.batch_size):
            batch = [sequences[i] for i in order[start : start + self.config.batch_size]]
            width = max(len(s) for s in batch)
            padded = torch.zeros(len(batch), width, dtype=torch.long)
            for i, seq in enumerate(batch):
                padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

            inputs, targets = padded[:, :-1], padded[:, 1:]
            logits, _ = self.model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,  # PAD
            )
            if train and optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimiser.step()
            total += float(loss.item())
            n_batches += 1
        return total / max(n_batches, 1)

    @staticmethod
    def prior_data_limitation(n_training: int) -> Dict[str, Any]:
        return {
            "moses_pretraining": "pending",
            "reason": (
                "MOSES (~1.9M molecules) is not bundled with this project. The prior "
                f"was trained on {n_training} locally available SMILES instead."
            ),
            "expected_effect": (
                "A character-level model trained on this little data learns SMILES "
                "grammar only partially, so validity rates are far below the >90% a "
                "MOSES-pretrained prior achieves. Measured validity is reported with "
                "every sampling run rather than assumed."
            ),
            "how_to_enable": "See scripts/download_moses.py.",
        }

    # -------------------------------------------------------------- sampling

    def sample(
        self,
        n: int = 100,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sample SMILES, then validate with RDKit and canonicalise.

        Returns valid molecules and the full honest accounting of what was
        rejected. Invalid strings are reported, never silently discarded.
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("MolecularGenerator.sample() called before train_prior().")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        sequences, log_probs = self._sample_sequences(
            n, temperature, max_length or self.config.max_length, random_state
        )
        raw = [self.vocab.decode(seq) for seq in sequences]

        valid, invalid = [], []
        for text in raw:
            if not text:
                invalid.append({"smiles": text, "reason": "empty string"})
                continue
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                invalid.append({"smiles": text, "reason": "RDKit could not parse"})
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                invalid.append({"smiles": text, "reason": f"sanitisation failed: {e}"})
                continue
            valid.append(Chem.MolToSmiles(mol, canonical=True))

        unique = list(dict.fromkeys(valid))
        novel = [s for s in unique if s not in self.training_smiles]

        return {
            "requested": n,
            "raw_generated": raw,
            "valid_smiles": valid,
            "unique_smiles": unique,
            "novel_smiles": novel,
            "invalid": invalid[:50],
            "n_invalid": len(invalid),
            "metrics": self.evaluate(valid, unique, novel, n),
            "log_probs": [round(float(v), 3) for v in log_probs],
            "temperature": temperature,
        }

    def _sample_sequences(
        self,
        n: int,
        temperature: float,
        max_length: int,
        random_state: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[float]]:
        if random_state is not None:
            torch.manual_seed(random_state)
        self.model.eval()

        bos, eos = self.vocab.stoi[BOS], self.vocab.stoi[EOS]
        tokens = torch.full((n, 1), bos, dtype=torch.long)
        hidden = None
        finished = torch.zeros(n, dtype=torch.bool)
        collected = [[] for _ in range(n)]
        total_log_prob = torch.zeros(n)

        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.model(tokens, hidden)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                log_prob = torch.log(probs.gather(1, nxt).squeeze(1) + 1e-12)

                total_log_prob += log_prob * (~finished).float()
                for i in range(n):
                    if not finished[i]:
                        collected[i].append(int(nxt[i, 0]))
                finished |= nxt.squeeze(1) == eos
                if bool(finished.all()):
                    break
                tokens = nxt

        return collected, total_log_prob.tolist()

    def sequence_log_prob(self, sequences: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of given token sequences under this model (differentiable)."""
        inputs, targets = sequences[:, :-1], sequences[:, 1:]
        logits, _ = self.model(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
        mask = (targets != 0).float()  # ignore PAD
        return (gathered * mask).sum(dim=1)

    # ------------------------------------------------------------ evaluation

    def _scaffold(self, smiles: str) -> str:
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
        except Exception:
            return ""

    def evaluate(
        self,
        valid: Sequence[str],
        unique: Optional[Sequence[str]] = None,
        novel: Optional[Sequence[str]] = None,
        n_requested: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Standard generative-model metrics, each defined explicitly because the
        denominators differ between papers.
        """
        unique = list(dict.fromkeys(valid)) if unique is None else list(unique)
        novel = (
            [s for s in unique if s not in self.training_smiles]
            if novel is None
            else list(novel)
        )
        n_requested = n_requested or len(valid)

        metrics: Dict[str, Any] = {
            "n_requested": int(n_requested),
            "n_valid": len(valid),
            "n_unique": len(unique),
            "n_novel": len(novel),
            "validity": round(len(valid) / max(n_requested, 1), 4),
            "uniqueness": round(len(unique) / max(len(valid), 1), 4),
            "novelty": round(len(novel) / max(len(unique), 1), 4),
            "definitions": {
                "validity": "valid / requested",
                "uniqueness": "unique valid / valid",
                "novelty": "unique and absent from the prior's training set / unique",
                "internal_diversity": "1 - mean pairwise ECFP4 Tanimoto among unique valid molecules",
            },
        }

        if len(unique) >= 2:
            fps = []
            for s in unique:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    fps.append(self.fp_gen.GetFingerprint(mol))
            if len(fps) >= 2:
                sims = []
                for i in range(len(fps)):
                    sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
                metrics["internal_diversity"] = round(1.0 - float(np.mean(sims)), 4)
                metrics["mean_pairwise_similarity"] = round(float(np.mean(sims)), 4)

            scaffolds = {self._scaffold(s) for s in unique}
            scaffolds.discard("")
            metrics["n_unique_scaffolds"] = len(scaffolds)
            metrics["scaffold_diversity"] = round(len(scaffolds) / max(len(unique), 1), 4)
            metrics["n_novel_scaffolds"] = len(scaffolds - self.training_scaffolds)

        return metrics

    # --------------------------------------------------------- reinforcement

    def reinforce(
        self,
        reward_fn: Callable[[List[str]], np.ndarray],
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        sigma: Optional[float] = None,
        diversity_filter: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        REINVENT augmented-likelihood RL.

        `reward_fn` maps a list of canonical SMILES to a reward array in [0, 1].
        The prior is frozen; the agent is a copy that is optimised towards

            log P_augmented = log P_prior + sigma * R
            Loss = (log P_augmented - log P_agent)^2

        Invalid generations receive reward 0, so the agent is penalised for them
        without being able to collapse onto a degenerate string: the prior term
        holds it to chemically plausible likelihoods.
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("reinforce() called before train_prior().")

        steps = steps or self.config.rl_steps
        batch_size = batch_size or self.config.rl_batch_size
        sigma = sigma if sigma is not None else self.config.rl_sigma
        started = time.time()

        import copy

        prior = copy.deepcopy(self.model)
        for p in prior.parameters():
            p.requires_grad_(False)
        prior.eval()

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config.rl_lr)
        scaffold_counts: Dict[str, int] = {}
        history: List[Dict[str, Any]] = []

        for step in range(steps):
            sequences, _ = self._sample_sequences(
                batch_size, 1.0, self.config.max_length, random_state=None
            )
            width = max((len(s) for s in sequences), default=1) + 1
            padded = torch.zeros(batch_size, width + 1, dtype=torch.long)
            padded[:, 0] = self.vocab.stoi[BOS]
            for i, seq in enumerate(sequences):
                if seq:
                    padded[i, 1 : 1 + len(seq)] = torch.tensor(seq, dtype=torch.long)

            decoded = [self.vocab.decode(seq) for seq in sequences]
            canonical: List[str] = []
            valid_mask = np.zeros(batch_size, dtype=bool)
            for i, text in enumerate(decoded):
                mol = Chem.MolFromSmiles(text) if text else None
                if mol is None:
                    canonical.append("")
                    continue
                canonical.append(Chem.MolToSmiles(mol, canonical=True))
                valid_mask[i] = True

            rewards = np.zeros(batch_size, dtype=np.float64)
            valid_smiles = [canonical[i] for i in range(batch_size) if valid_mask[i]]
            if valid_smiles:
                scored = np.asarray(reward_fn(valid_smiles), dtype=np.float64)
                if scored.shape[0] != len(valid_smiles):
                    raise ValueError("reward_fn must return one reward per SMILES.")
                rewards[valid_mask] = scored

            # Diversity filter: repeated scaffolds earn progressively less, so
            # the agent cannot farm reward from a single good molecule.
            if diversity_filter:
                for i in range(batch_size):
                    if not valid_mask[i]:
                        continue
                    scaf = self._scaffold(canonical[i])
                    if not scaf:
                        continue
                    seen = scaffold_counts.get(scaf, 0)
                    scaffold_counts[scaf] = seen + 1
                    if seen:
                        rewards[i] *= 1.0 / (1.0 + seen)

            reward_t = torch.tensor(rewards, dtype=torch.float32)
            agent_lp = self.sequence_log_prob(padded)
            with torch.no_grad():
                inputs, targets = padded[:, :-1], padded[:, 1:]
                prior_logits, _ = prior(inputs)
                prior_log = F.log_softmax(prior_logits, dim=-1)
                prior_lp = (
                    prior_log.gather(2, targets.unsqueeze(2)).squeeze(2)
                    * (targets != 0).float()
                ).sum(dim=1)

            augmented = prior_lp + sigma * reward_t
            loss = torch.mean((augmented - agent_lp) ** 2)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimiser.step()

            entry = {
                "step": step,
                "loss": round(float(loss.item()), 3),
                "mean_reward": round(float(rewards.mean()), 4),
                "mean_reward_valid": (
                    round(float(rewards[valid_mask].mean()), 4) if valid_mask.any() else 0.0
                ),
                "validity": round(float(valid_mask.mean()), 4),
                "n_unique_scaffolds": len(scaffold_counts),
            }
            history.append(entry)
            if verbose and step % 10 == 0:
                print(
                    f"  step {step:3d} loss {entry['loss']:9.1f} "
                    f"reward {entry['mean_reward']:.3f} validity {entry['validity']:.2f}"
                )

        first = history[0] if history else {}
        last = history[-1] if history else {}
        self.status = "rl_trained"
        self.rl_report = {
            "status": "trained",
            "method": "REINVENT augmented likelihood (Olivecrona et al. 2017)",
            "objective": "Loss = (log P_prior + sigma * R - log P_agent)^2",
            "sigma": sigma,
            "steps": steps,
            "batch_size": batch_size,
            "diversity_filter": diversity_filter,
            "seconds": round(time.time() - started, 1),
            "initial_mean_reward": first.get("mean_reward"),
            "final_mean_reward": last.get("mean_reward"),
            "initial_validity": first.get("validity"),
            "final_validity": last.get("validity"),
            "n_unique_scaffolds_seen": len(scaffold_counts),
            "history": history,
            "caveats": [
                "The prior was trained on ~2,500 local SMILES, not MOSES, so "
                "absolute validity is low; the RL loop optimises what the prior "
                "can express.",
                "Reward uses the LOWER conformal bound, so the agent is rewarded "
                "for confident predictions rather than optimistic ones.",
                "Reward improvement is improvement against this platform's own "
                "models. It is not experimental validation of the molecules.",
            ],
        }
        return self.rl_report

    # --------------------------------------------------------- persistence

    def save(self, path: str):
        if self.model is None or self.vocab is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "config": self.config.as_dict(),
                "vocab": self.vocab.as_dict(),
                "state_dict": self.model.state_dict(),
                "training_smiles": sorted(self.training_smiles),
                "training_scaffolds": sorted(self.training_scaffolds),
                "prior_report": self.prior_report,
                "rl_report": self.rl_report,
                "status": self.status,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "MolecularGenerator":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        gen = cls(GeneratorConfig(**blob["config"]))
        gen.vocab = SmilesVocabulary.from_dict(blob["vocab"])
        gen.model = SmilesRNN(len(gen.vocab), gen.config)
        gen.model.load_state_dict(blob["state_dict"])
        gen.training_smiles = set(blob["training_smiles"])
        gen.training_scaffolds = set(blob["training_scaffolds"])
        gen.prior_report = blob.get("prior_report", {})
        gen.rl_report = blob.get("rl_report", {})
        gen.status = blob.get("status", "prior_trained")
        return gen

    # RDKit's FingerprintGenerator64 is a Boost.Python object and cannot be
    # pickled or deep-copied. It is fully determined by its parameters, so it is
    # dropped on the way out and rebuilt on the way in.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("fp_gen", None)
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
        self.fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def summary(self) -> Dict[str, Any]:
        if self.model is None:
            return {
                "status": "not_trained",
                "reason": "The generator prior has not been trained in this process.",
                "moses_pretraining": "pending",
            }
        return {
            "status": self.status,
            "architecture": {
                "type": "character-level GRU SMILES language model",
                "embedding_size": self.config.embedding_size,
                "hidden_size": self.config.hidden_size,
                "n_layers": self.config.n_layers,
                "vocabulary_size": len(self.vocab) if self.vocab else 0,
            },
            "prior": self.prior_report,
            "reinforcement_learning": self.rl_report or {
                "status": "not_run",
                "reason": "No RL campaign has been run in this process.",
            },
        }
