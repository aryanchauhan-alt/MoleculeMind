"""Shared fixtures. The ModelService is expensive to build, so it is session-scoped."""

import os
import sys
import warnings
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="session")
def data_dir():
    return os.path.join(ROOT, "backend", "data")


@pytest.fixture(scope="session")
def dataset_manager(data_dir):
    from backend.app.core.dataset import DatasetManager

    return DatasetManager(data_dir=data_dir)


@pytest.fixture(scope="session")
def model_service(data_dir):
    """A quick-configuration ModelService (fewer trees) shared across the suite."""
    from backend.app.core.models import ModelService

    svc = ModelService(data_dir=data_dir, quick=True)
    svc.initialize()
    return svc


@pytest.fixture(scope="session")
def al_engine(model_service):
    from backend.app.core.active_learning import ActiveLearningEngine

    return ActiveLearningEngine(
        dataset_manager=model_service.dataset_manager,
        pool_payload=model_service.al_pool_payload,
    )
