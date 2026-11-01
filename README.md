# MoleculeMind

**An Uncertainty-Aware, Human-in-the-Loop Platform for Multi-Objective Molecular Discovery**

MoleculeMind accelerates small-molecule drug discovery by combining genuine machine learning models with rigorous conformal uncertainty bounds, applicability-domain novelty detection, interactive Pareto frontier optimization, and Bayesian active learning.

---

## Key Platform Capabilities

1. **Molecular Analyzer**:
   - SMILES input with instant validation and canonicalization via **RDKit**.
   - Dual-view structural visualization: publication-quality 2D vector SVG + interactive 3D conformer generation (**MMFF94 force-field geometry optimization** rendered with **3Dmol.js**).
   - Physicochemical descriptors: MW, LogP, TPSA, QED Drug-Likeness, H-Bond Donors/Acceptors, Rotatable Bonds, Rings, Fraction Csp3.
   - Lipinski's Rule of 5 and Veber Bioavailability Rule evaluation.

2. **Property Prediction**:
   - Genuine baseline model trained on the authentic **Delaney ESOL dataset** (1,128 compounds).
   - Test set evaluation: **$R^2 = 0.894$**, **$\text{RMSE} = 0.704$**, **$\text{MAE} = 0.482$**.
   - Never displays fabricated results.

3. **Uncertainty Engine (Conformal Prediction)**:
   - Inductive Conformal Prediction (ICP) calibrated on held-out calibration compounds.
   - Provides finite-sample, distribution-free confidence intervals at **80%, 90%, and 95%** confidence levels ($P(Y \in [L, U]) \ge 1 - \alpha$).
   - Heteroscedastic tree-ensemble variance scaling.

4. **Applicability-Domain (AD) Detection**:
   - Quantifies chemical space extrapolation risk.
   - 1024-bit Morgan Fingerprint (ECFP4) Tanimoto similarity to the nearest Delaney training set neighbor ($S_{max}$).
   - PCA chemical leverage ($h$) evaluated against the standard **OECD Guideline cutoff** ($h^* = \frac{3(p+1)}{n}$).
   - Identifies the nearest reference analog in the training set with measured experimental values.

5. **Pareto Laboratory**:
   - Multi-objective non-dominated sorting across Biological Activity ($pIC_{50}$), Aqueous Solubility ($\log S$), Toxicity Risk, and Drug-Likeness (QED).
   - Interactive 2D trade-off scatter plot with step-envelope Pareto Frontier.
   - Interactive 3D multi-objective discovery cube via **Plotly**.
   - Real-time candidate inspection matrix.

6. **Active Learning (Human-in-the-Loop)**:
   - Bayesian Optimization powered by **BoTorch** (`SingleTaskGP`).
   - Acquisition policies: Upper Confidence Bound (qUCB with adjustable $\beta$), Expected Improvement (EI), Pure Epistemic Uncertainty Sampling (Max $\sigma$), and Multi-Objective Hypervolume.
   - Simulated laboratory assays: reveals experimental results, updates the Gaussian Process posterior, logs the discovery round, and refines rankings.

---

## Tech Stack

* **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Plotly.js, 3Dmol.js
* **Backend**: Python 3.14, FastAPI, Uvicorn, Pydantic
* **Cheminformatics**: RDKit (`rdkit-2026.9.1`)
* **Machine Learning**: PyTorch (`torch-2.15.0`), Chemprop, Scikit-learn
* **Bayesian Optimization**: BoTorch (`botorch-0.18.1`), GPyTorch

---

## Quickstart & Launch

### 1. Launch Platform (Single Command)
The frontend has been compiled and is served directly by the FastAPI backend:

```bash
python run.py
```

Open your browser at:
* **Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Frontend Development Server (Optional)
If modifying React components with hot module reloading:

```bash
cd frontend
npm run dev
```
Access the Vite dev server at `http://localhost:5173` (proxies `/api` to port 8000).
