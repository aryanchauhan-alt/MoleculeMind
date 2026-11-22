"""
Central configuration.

Every tunable the platform has is declared here and can be overridden by an
environment variable prefixed `MOLECULEMIND_`. Nothing in the scientific code
should contain a magic number that is not either derived from data or declared
here, so that a run's configuration can be recorded alongside its metrics.
"""

import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Tuple


def _env_str(key: str, default: str) -> str:
    return os.environ.get(f"MOLECULEMIND_{key}", default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(f"MOLECULEMIND_{key}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(f"MOLECULEMIND_{key}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(f"MOLECULEMIND_{key}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))


@dataclass(frozen=True)
class Settings:
    # --- paths ----------------------------------------------------------
    data_dir: str = field(default_factory=lambda: _env_str(
        "DATA_DIR", os.path.join(_project_root(), "backend", "data")
    ))
    artifact_dir: str = field(default_factory=lambda: _env_str(
        "ARTIFACT_DIR", os.path.join(_project_root(), "backend", "artifacts")
    ))

    # --- reproducibility -------------------------------------------------
    random_seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    # --- model sizing (reduce for a fast smoke run) ----------------------
    esol_n_estimators: int = field(default_factory=lambda: _env_int("ESOL_TREES", 150))
    esol_max_depth: int = field(default_factory=lambda: _env_int("ESOL_MAX_DEPTH", 20))
    clintox_n_estimators: int = field(default_factory=lambda: _env_int("CLINTOX_TREES", 300))

    # --- data partitioning -----------------------------------------------
    test_fraction: float = field(default_factory=lambda: _env_float("TEST_FRACTION", 0.15))
    cal_fraction: float = field(default_factory=lambda: _env_float("CAL_FRACTION", 0.15))
    al_pool_size: int = field(default_factory=lambda: _env_int("AL_POOL_SIZE", 30))
    al_seed_size: int = field(default_factory=lambda: _env_int("AL_SEED_SIZE", 6))

    # --- conformal prediction --------------------------------------------
    # Confidence levels to precompute. alpha = 1 - confidence.
    conformal_levels: Tuple[float, ...] = (0.80, 0.90, 0.95)

    # --- applicability domain --------------------------------------------
    # Thresholds are DERIVED from training-set diagnostics at fit time (see
    # ApplicabilityDomainEngine.fit); these percentiles control that derivation
    # rather than hardcoding a similarity cut-off.
    ad_similarity_percentile: float = field(
        default_factory=lambda: _env_float("AD_SIM_PERCENTILE", 5.0)
    )
    ad_distance_percentile: float = field(
        default_factory=lambda: _env_float("AD_DIST_PERCENTILE", 95.0)
    )
    ad_pca_components: int = field(default_factory=lambda: _env_int("AD_PCA_COMPONENTS", 10))

    # --- D-MPNN ensemble (Stage 3) ---------------------------------------
    # Training five members costs ~70 s on CPU; the checkpoint is cached, so
    # only the first run pays. Set false to skip it entirely.
    enable_dmpnn: bool = field(default_factory=lambda: _env_bool("ENABLE_DMPNN", True))
    dmpnn_members: int = field(default_factory=lambda: _env_int("DMPNN_MEMBERS", 5))
    dmpnn_hidden_size: int = field(default_factory=lambda: _env_int("DMPNN_HIDDEN", 300))
    dmpnn_depth: int = field(default_factory=lambda: _env_int("DMPNN_DEPTH", 3))
    dmpnn_max_epochs: int = field(default_factory=lambda: _env_int("DMPNN_EPOCHS", 80))
    # Use the D-MPNN's learned embeddings as the applicability-domain latent space.
    use_latent_ad: bool = field(default_factory=lambda: _env_bool("USE_LATENT_AD", True))

    # --- Stage 4: probabilistic Pareto, risk slider, planner, generator ---
    pareto_mc_samples: int = field(default_factory=lambda: _env_int("PARETO_SAMPLES", 200))
    pareto_prefilter_k: float = field(default_factory=lambda: _env_float("PARETO_PREFILTER_K", 4.0))
    risk_kappa_max: float = field(default_factory=lambda: _env_float("RISK_KAPPA_MAX", 2.0))
    ehvi_samples: int = field(default_factory=lambda: _env_int("EHVI_SAMPLES", 128))
    # The generator prior costs ~75 s to train; the checkpoint is cached.
    enable_generator: bool = field(default_factory=lambda: _env_bool("ENABLE_GENERATOR", True))

    # --- active learning --------------------------------------------------
    al_pca_components: int = field(default_factory=lambda: _env_int("AL_PCA_COMPONENTS", 8))
    al_default_beta: float = field(default_factory=lambda: _env_float("AL_BETA", 2.0))
    # Refit conformal calibration once this many new labels have arrived.
    al_recalibration_interval: int = field(
        default_factory=lambda: _env_int("AL_RECALIBRATION_INTERVAL", 1)
    )

    # --- persistence ------------------------------------------------------
    # When true, a matching cached model is loaded instead of retraining.
    use_model_cache: bool = field(default_factory=lambda: _env_bool("USE_MODEL_CACHE", True))
    write_model_cache: bool = field(default_factory=lambda: _env_bool("WRITE_MODEL_CACHE", True))

    # --- api --------------------------------------------------------------
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    cors_origins: Tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    )

    def as_dict(self) -> Dict[str, Any]:
        """Serialisable record of the configuration a run used."""
        d = asdict(self)
        d["conformal_levels"] = list(self.conformal_levels)
        d["cors_origins"] = list(self.cors_origins)
        return d

    def training_fingerprint(self) -> Dict[str, Any]:
        """The subset of settings that affects trained-model identity."""
        return {
            "random_seed": self.random_seed,
            "esol_n_estimators": self.esol_n_estimators,
            "esol_max_depth": self.esol_max_depth,
            "clintox_n_estimators": self.clintox_n_estimators,
            "test_fraction": self.test_fraction,
            "cal_fraction": self.cal_fraction,
            "al_pool_size": self.al_pool_size,
            "ad_similarity_percentile": self.ad_similarity_percentile,
            "ad_distance_percentile": self.ad_distance_percentile,
            "ad_pca_components": self.ad_pca_components,
            "enable_dmpnn": self.enable_dmpnn,
            "dmpnn_members": self.dmpnn_members,
            "dmpnn_hidden_size": self.dmpnn_hidden_size,
            "dmpnn_depth": self.dmpnn_depth,
            "dmpnn_max_epochs": self.dmpnn_max_epochs,
            "use_latent_ad": self.use_latent_ad,
            "enable_generator": self.enable_generator,
        }


settings = Settings()
