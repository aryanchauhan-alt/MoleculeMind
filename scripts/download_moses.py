"""
How to obtain MOSES for generator pre-training.

MOSES (Molecular Sets, Polykovskiy et al. 2020) is a ~1.9M molecule benchmark
dataset curated from ZINC. It is NOT bundled with this project and is NOT
downloaded automatically: it is ~130 MB and the project must remain runnable
offline with no implicit network access.

Why it matters
--------------
The generator's prior is currently trained on ~2,500 locally available SMILES
(Delaney ESOL + ClinTox). A character-level language model trained on that much
data learns SMILES grammar only partially, and the measured validity rate
(~45-60% at temperature 1.0) reflects that. A MOSES-pretrained prior of the same
architecture typically reaches >90% validity.

Obtaining the data
------------------
Option 1 - the official package:

    pip install molsets
    python -c "import moses; moses.get_dataset('train')"

Option 2 - direct download of the split files:

    https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv
    https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv

Place `train.csv` (a single `SMILES` column) at:

    backend/data/moses_train.csv

Using it
--------
Once the file is present, train the prior on it:

    from backend.app.core.generator import GeneratorConfig, MolecularGenerator
    import pandas as pd

    smiles = pd.read_csv("backend/data/moses_train.csv")["SMILES"].tolist()
    gen = MolecularGenerator(GeneratorConfig(hidden_size=512, n_layers=3, max_epochs=20))
    gen.train_prior(smiles, verbose=True)
    gen.save("backend/artifacts/generator-moses.pt")

Expect several hours on CPU for the full set; a 200k-molecule subset trains in
well under an hour and already lifts validity substantially. Nothing else in the
pipeline changes - the RL loop, the reward and the evaluation metrics are
unchanged, which is the point of keeping the prior swappable.

Citation
--------
Polykovskiy et al. (2020), "Molecular Sets (MOSES): A Benchmarking Platform for
Molecular Generation Models", Frontiers in Pharmacology.
"""

import os
import sys

TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "data", "moses_train.csv",
)

if __name__ == "__main__":
    if os.path.exists(TARGET):
        print(f"MOSES training data already present at {TARGET}")
        sys.exit(0)
    print(__doc__)
    print(f"\nExpected location: {TARGET}")
    print("This script intentionally does not download anything by itself.")
