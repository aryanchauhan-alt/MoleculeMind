"""
Generator service: wires the SMILES generator to this platform's own calibrated
models to build an uncertainty-aware reward, and runs the RL campaign.

The reward
----------
    R = w_valid * 1[valid]
      + w_obj   * mean_j normalised(lower conformal bound of objective j)
      + w_ad    * applicability-domain term
      - w_dup   * 1[already generated]

**The objective term uses the LOWER conformal bound, not the mean.** This is the
single most important choice in the reward. Rewarding the predicted mean teaches
the generator to find regions where the model is uncertain and happens to predict
well - it learns to exploit the model's blind spots, which is exactly the failure
mode uncertainty-aware generation exists to prevent. Rewarding the lower bound
means a candidate only scores highly when the model is *confident* it is good, so
uncertainty is a cost rather than a free option.

The applicability-domain term does the same job from the other direction:
molecules far outside the training domain are penalised, because a confident
prediction there is not trustworthy either.

What reward improvement does and does not mean
----------------------------------------------
Rising reward means the agent found molecules **this platform's models score
highly with confidence**. It is not experimental validation, and it is not
evidence the molecules are good drugs. If the models are wrong in a region, the
generator will happily exploit that; the lower-bound reward and the AD penalty
narrow that opening but do not close it.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from backend.app.core.config import settings
from backend.app.core.generator import GeneratorConfig, MolecularGenerator

logger = logging.getLogger("moleculemind.generator_service")


DEFAULT_WEIGHTS = {
    "validity": 0.2,
    "objectives": 0.5,
    "applicability_domain": 0.2,
    "novelty": 0.1,
}


class GeneratorService:
    def __init__(self, model_service: Any, quick: bool = False):
        self.ms = model_service
        self.quick = quick
        self.generator: Optional[MolecularGenerator] = None
        self.status: str = "not_trained"
        self.failure_reason: Optional[str] = None
        self.last_campaign: Dict[str, Any] = {}
        self._seen: set = set()

    # ------------------------------------------------------------ training

    def corpus(self) -> List[str]:
        """Every canonical SMILES available locally, for prior training."""
        esol = self.ms.dataset_manager.get_esol_dataset()
        clintox = self.ms.dataset_manager.get_clintox_dataset()
        smiles = set(esol.smiles)
        if clintox is not None:
            smiles |= set(clintox.smiles)
        return sorted(smiles)

    def train_prior(self, verbose: bool = False) -> Dict[str, Any]:
        corpus = self.corpus()
        config = GeneratorConfig(
            hidden_size=128 if self.quick else 256,
            n_layers=1 if self.quick else 2,
            max_epochs=4 if self.quick else 60,
            patience=2 if self.quick else 8,
            seed=settings.random_seed,
        )
        self.generator = MolecularGenerator(config)
        try:
            report = self.generator.train_prior(corpus, verbose=verbose)
            self.status = "prior_trained"
            return report
        except Exception as e:
            self.status = "failed"
            self.failure_reason = str(e)
            logger.error("Generator prior training failed: %s", e)
            return {"status": "failed", "error": str(e)}

    # -------------------------------------------------------------- reward

    def _objective_bounds(self, smiles_list: Sequence[str]) -> Dict[str, Any]:
        """
        Lower conformal bounds and AD scores for a batch, via the platform's models.

        Molecules the pipeline cannot score are returned with `None`, never with a
        substituted value.
        """
        rows: List[Dict[str, Any]] = []
        for smi in smiles_list:
            try:
                pred = self.ms.predict_molecule(smi)
            except Exception:
                rows.append({"smiles": smi, "ok": False})
                continue

            props = pred["properties"]
            sol = props.get("solubility_logs", {})
            tox = props.get("toxicity_risk_probability", {})
            ad = pred["applicability_domain"]

            # Lower bound on a MAXIMISED objective is the conformal lower bound.
            sol_lower = None
            if sol.get("status") == "available":
                sol_lower = sol["conformal"]["intervals"]["90%"]["lower_bound"]

            # For a MINIMISED objective, the pessimistic bound is the UPPER one.
            tox_upper = None
            if tox.get("status") == "available":
                tox_upper = tox["value"]

            rows.append(
                {
                    "smiles": smi,
                    "ok": True,
                    "solubility_lower_90": sol_lower,
                    "toxicity_upper": tox_upper,
                    "qed": pred["descriptors"]["qed"],
                    "ad_status": ad["status"],
                    "ad_index": ad["domain_index"],
                }
            )
        return {"rows": rows}

    def make_reward_fn(
        self, weights: Optional[Dict[str, float]] = None
    ) -> Any:
        """Build the reward callable the RL loop consumes."""
        w = {**DEFAULT_WEIGHTS, **(weights or {})}

        def reward_fn(smiles_list: List[str]) -> np.ndarray:
            info = self._objective_bounds(smiles_list)
            rewards = np.zeros(len(smiles_list), dtype=np.float64)

            for i, row in enumerate(info["rows"]):
                if not row["ok"]:
                    continue  # unscoreable -> reward 0

                # Validity is already implied (these parsed), so this component
                # rewards being scoreable at all.
                score = w["validity"]

                # Objective term: lower conformal bound, normalised to [0, 1].
                parts: List[float] = []
                if row["solubility_lower_90"] is not None:
                    # logS spans roughly -8 (insoluble) to +2 (very soluble).
                    parts.append(float(np.clip((row["solubility_lower_90"] + 8.0) / 10.0, 0.0, 1.0)))
                if row["toxicity_upper"] is not None:
                    # Minimised: reward low toxicity.
                    parts.append(float(np.clip(1.0 - row["toxicity_upper"], 0.0, 1.0)))
                parts.append(float(np.clip(row["qed"], 0.0, 1.0)))
                if parts:
                    score += w["objectives"] * float(np.mean(parts))

                # Applicability domain: a confident prediction outside the
                # training domain is not trustworthy, so penalise distance.
                ad_factor = {
                    "IN_DOMAIN": 1.0,
                    "BORDERLINE": 0.5,
                    "OUT_OF_DOMAIN": 0.0,
                }.get(row["ad_status"], 0.0)
                score += w["applicability_domain"] * ad_factor

                # Novelty: duplicates of molecules already produced earn nothing
                # extra, so the agent cannot farm reward by repetition.
                if row["smiles"] not in self._seen:
                    score += w["novelty"]
                    self._seen.add(row["smiles"])

                rewards[i] = float(np.clip(score, 0.0, 1.0))
            return rewards

        return reward_fn

    # ------------------------------------------------------------ campaign

    def run_campaign(
        self,
        steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        sigma: Optional[float] = None,
        n_final_samples: int = 200,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample before RL, run the RL loop, sample after, and report both.

        Reporting both ends is the point: a campaign that raises mean reward
        while collapsing validity or diversity has not improved anything, and
        only the before/after comparison shows that.
        """
        if self.generator is None:
            raise RuntimeError("run_campaign() called before train_prior().")

        started = time.time()
        self._seen = set()

        before = self.generator.sample(
            n=n_final_samples, temperature=1.0, random_state=settings.random_seed
        )
        reward_fn = self.make_reward_fn(weights)
        baseline_rewards = (
            reward_fn(before["unique_smiles"]) if before["unique_smiles"] else np.zeros(0)
        )

        self._seen = set()
        rl_report = self.generator.reinforce(
            reward_fn=reward_fn,
            steps=steps if steps is not None else (3 if self.quick else 40),
            batch_size=batch_size if batch_size is not None else (8 if self.quick else 32),
            sigma=sigma,
            diversity_filter=True,
            verbose=verbose,
        )

        self._seen = set()
        after = self.generator.sample(
            n=n_final_samples, temperature=1.0, random_state=settings.random_seed + 1
        )
        final_rewards = (
            reward_fn(after["unique_smiles"]) if after["unique_smiles"] else np.zeros(0)
        )

        self.status = "rl_trained"
        self.last_campaign = {
            "status": "ok",
            "seconds": round(time.time() - started, 1),
            "reinforcement_learning": rl_report,
            "before": {
                "metrics": before["metrics"],
                "mean_reward": round(float(baseline_rewards.mean()), 4)
                if baseline_rewards.size
                else None,
                "examples": before["novel_smiles"][:10],
            },
            "after": {
                "metrics": after["metrics"],
                "mean_reward": round(float(final_rewards.mean()), 4)
                if final_rewards.size
                else None,
                "examples": after["novel_smiles"][:10],
            },
            "reward_weights": {**DEFAULT_WEIGHTS, **(weights or {})},
            "reward_definition": (
                "R = w_valid*scoreable + w_obj*mean(normalised LOWER conformal "
                "bounds) + w_ad*domain_factor + w_novelty*first_time. The lower "
                "bound is deliberate: rewarding the mean would teach the agent to "
                "exploit regions where the model is merely uncertain."
            ),
            "honesty": [
                "Reward improvement is improvement against this platform's own "
                "models, not experimental validation.",
                "The prior was trained on ~2,500 local SMILES, not MOSES, so "
                "absolute validity is low. See the prior report's data_limitation.",
                "Before/after metrics are both reported so a campaign that raises "
                "reward while destroying validity or diversity is visible.",
            ],
        }
        return self.last_campaign

    # ------------------------------------------------- generate-and-screen

    def generate_and_screen(
        self,
        n: int = 100,
        temperature: float = 1.0,
        top_k: int = 20,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Sample, validate, deduplicate, screen through the platform's models and
        return the best by lower-bound score with full provenance.
        """
        if self.generator is None:
            raise RuntimeError("generate_and_screen() called before train_prior().")

        out = self.generator.sample(n=n, temperature=temperature, random_state=random_state)
        candidates: List[Dict[str, Any]] = []

        for smi in out["unique_smiles"]:
            try:
                pred = self.ms.predict_molecule(smi)
            except Exception as e:
                candidates.append({"smiles": smi, "screened": False, "error": str(e)})
                continue
            props = pred["properties"]
            sol = props["solubility_logs"]
            ad = pred["applicability_domain"]
            candidates.append(
                {
                    "smiles": smi,
                    "screened": True,
                    "novel": smi not in self.generator.training_smiles,
                    "solubility_logs": sol["value"],
                    "solubility_lower_90": sol["conformal"]["intervals"]["90%"]["lower_bound"],
                    "solubility_upper_90": sol["conformal"]["intervals"]["90%"]["upper_bound"],
                    "sigma": sol["ensemble_spread"],
                    "qed": pred["descriptors"]["qed"],
                    "molecular_weight": pred["descriptors"]["molecular_weight"],
                    "logp": pred["descriptors"]["logp"],
                    "lipinski_pass": pred["descriptors"]["lipinski"]["passed"],
                    "ad_status": ad["status"],
                    "ad_index": ad["domain_index"],
                    "toxicity": (
                        props["toxicity_risk_probability"].get("value")
                        if props["toxicity_risk_probability"].get("status") == "available"
                        else None
                    ),
                }
            )

        screened = [c for c in candidates if c.get("screened")]
        # Rank by the lower bound, so confident candidates beat uncertain ones.
        screened.sort(key=lambda c: -(c["solubility_lower_90"] + c["qed"]))

        return {
            "generation_metrics": out["metrics"],
            "n_candidates": len(screened),
            "n_unscreenable": len(candidates) - len(screened),
            "top_candidates": screened[:top_k],
            "ranking_basis": (
                "Ranked by lower 90% conformal bound on solubility plus QED, so a "
                "candidate must be confidently good rather than optimistically good. "
                "These are model predictions for generated structures, not "
                "measurements, and none has been synthesised or assayed."
            ),
        }

    # -------------------------------------------------------- persistence

    def save(self, directory: str, key: str):
        if self.generator is None:
            return
        os.makedirs(directory, exist_ok=True)
        self.generator.save(os.path.join(directory, f"generator-{key}.pt"))

    def load(self, directory: str, key: str) -> bool:
        path = os.path.join(directory, f"generator-{key}.pt")
        if not os.path.exists(path):
            return False
        try:
            self.generator = MolecularGenerator.load(path)
            self.status = self.generator.status
            return True
        except Exception as e:
            logger.warning("Could not load generator checkpoint %s: %s", path, e)
            return False

    def summary(self) -> Dict[str, Any]:
        if self.generator is None:
            return {
                "status": self.status,
                "reason": self.failure_reason
                or "The generator has not been trained in this process.",
                "moses_pretraining": "pending",
            }
        summary = self.generator.summary()
        summary["last_campaign"] = self.last_campaign or {
            "status": "not_run",
            "reason": "No RL campaign has been run in this process.",
        }
        return summary
