import React, { useState } from 'react';
import { Info, HelpCircle, X, BookOpen } from 'lucide-react';

export interface ScientificConcept {
  term: string;
  category: 'Structure' | 'Prediction' | 'Uncertainty' | 'Optimization' | 'Active Learning';
  summary: string;
  simpleExplanation: string;
  scientificFormula?: string;
  reference?: string;
}

export const SCIENTIFIC_GLOSSARY: Record<string, ScientificConcept> = {
  smiles: {
    term: 'SMILES (Simplified Molecular Input Line Entry System)',
    category: 'Structure',
    summary: 'A standard chemical specification notation using short ASCII character strings.',
    simpleExplanation: 'A compact text code that describes chemical bonds, atoms, rings and branches. For example, CC(=O)Oc1ccccc1C(=O)O represents Aspirin.',
    scientificFormula: 'Graph G = (V, E) serialized via depth-first traversal with ring-closure digits',
    reference: 'Weininger, D. (1988). SMILES, a chemical language and information system.'
  },
  pic50: {
    term: 'pIC50 (Biological Affinity Potency)',
    category: 'Prediction',
    summary: 'The negative logarithm of the half-maximal inhibitory concentration (IC50) in molar units.',
    simpleExplanation: 'Measures how potently a compound inhibits its biological drug target. Higher numbers indicate higher potency: a pIC50 of 6 means micromolar potency (1 µM), while 9 means nanomolar potency (1 nM).',
    scientificFormula: 'pIC50 = -log10(IC50 in M)',
    reference: 'Cheng & Prusoff (1973) / ChEMBL potency standards'
  },
  solubility: {
    term: 'Aqueous Solubility (logS)',
    category: 'Prediction',
    summary: 'The decimal logarithm of water solubility measured in mol/L.',
    simpleExplanation: 'Indicates how easily a drug dissolves in water and bloodstream fluids. Values above -2.0 are highly soluble; -2.0 to -4.0 are moderately soluble; below -4.0 are poorly soluble and may suffer poor absorption.',
    scientificFormula: 'logS = log10(Solubility in mol/L)',
    reference: 'Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.'
  },
  toxicity: {
    term: 'Toxicity Risk Probability',
    category: 'Prediction',
    summary: 'Machine learning probability estimate of clinical toxicity and trial failure.',
    simpleExplanation: 'Calculates the likelihood that a compound fails human clinical trials due to safety concerns. Lower is preferable (< 20% indicates favorable safety margin).',
    scientificFormula: 'P(Toxicity = 1 | ECFP4, Descriptors) ∈ [0, 1]',
    reference: 'ClinTox Benchmark / Tox21 Challenge'
  },
  qed: {
    term: 'QED (Quantitative Estimate of Drug-Likeness)',
    category: 'Structure',
    summary: 'Geometric mean of desirability functions combining 8 fundamental physicochemical properties.',
    simpleExplanation: 'A composite score between 0 and 1 assessing how much a compound looks like approved oral medicines. Above 0.67 is considered drug-like.',
    scientificFormula: 'QED = exp( (1/8) · ∑ ln d_i(x_i) )',
    reference: 'Bickerton et al. (2012). Quantifying the chemical beauty of drugs. Nature Chemistry.'
  },
  conformal: {
    term: 'Inductive Conformal Prediction (ICP)',
    category: 'Uncertainty',
    summary: 'A distribution-free statistical framework that outputs valid prediction intervals with mathematical finite-sample guarantees.',
    simpleExplanation: 'Instead of guessing a single number, the model provides a confidence range (e.g. 90% confidence that true solubility lies between -1.8 and -1.1). Guaranteed to hold regardless of data distribution.',
    scientificFormula: 'P( Y_new ∈ [ ŷ_new - q_{1-α}, ŷ_new + q_{1-α} ] ) ≥ 1 - α',
    reference: 'Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.'
  },
  uncertainty: {
    term: 'Prediction Uncertainty (Ensemble Variance σ)',
    category: 'Uncertainty',
    summary: 'Standard deviation across ensemble decision trees or Gaussian Process posterior variance.',
    simpleExplanation: 'Reflects how confident the AI model is about a specific molecule. Low σ means the trees agree closely; high σ indicates novel chemistry that the model has rarely seen.',
    scientificFormula: 'σ^2(x) = (1/T) · ∑ (f_t(x) - f̄(x))^2',
    reference: 'Efron, B. (1982). The Jackknife, the Bootstrap, and Other Resampling Plans.'
  },
  applicability_domain: {
    term: 'Applicability Domain (AD)',
    category: 'Uncertainty',
    summary: 'The chemical structural space in which the model makes mathematically reliable predictions without uncalibrated extrapolation.',
    simpleExplanation: 'Determines if a target molecule is structurally similar enough to the training set molecules to trust the AI output. Categorized into In Domain, Borderline, or Extrapolation.',
    scientificFormula: 'AD = { x : Tanimoto(x, x_train) ≥ threshold ∧ h(x) ≤ h* }',
    reference: 'OECD Guidelines for the Validation of (Q)SAR Models (2007).'
  },
  tanimoto: {
    term: 'Tanimoto Similarity (ECFP4 Fingerprints)',
    category: 'Structure',
    summary: 'Jaccard similarity coefficient between circular topological chemical fingerprints.',
    simpleExplanation: 'Measures structural overlap between two molecules from 0% (completely different) to 100% (identical circular substructure patterns). Over 60% indicates a close chemical analog.',
    scientificFormula: 'T(A, B) = |A ∩ B| / |A ∪ B|',
    reference: 'Rogers & Hahn (2010). Extended-Connectivity Fingerprints.'
  },
  leverage: {
    term: 'Chemical Leverage (Hat Matrix h)',
    category: 'Uncertainty',
    summary: 'Mahalanobis-based distance of a query compound from the training set centroid in descriptor space.',
    simpleExplanation: 'Identifies whether a molecule is an outlier in molecular weight, charge, or polarity. If leverage h exceeds the warning threshold h* = 3(p+1)/n, the prediction is an extrapolation.',
    scientificFormula: 'h = x^T (X^T X)^(-1) x, with warning cutoff h* = 3(p+1)/n',
    reference: 'Williams Plot methodology (Gramatica, 2007).'
  },
  pareto: {
    term: 'Pareto Optimality & Non-Dominated Frontier',
    category: 'Optimization',
    summary: 'A state where no objective can be improved without degrading at least one other objective.',
    simpleExplanation: 'In drug discovery, you cannot maximize potency, solubility, and safety all at once without trade-offs. Pareto-optimal molecules are the best compromises where no other compound beats them in every single category.',
    scientificFormula: 'x* ∈ P iff ∄ y : ∀i (f_i(y) ≥ f_i(x*)) ∧ ∃j (f_j(y) > f_j(x*))',
    reference: 'Vilfredo Pareto (1906) / Deb, K. (2001). Multi-Objective Optimization.'
  },
  acquisition: {
    term: 'Acquisition Functions (Bayesian Optimization)',
    category: 'Active Learning',
    summary: 'Mathematical heuristics that balance exploration and exploitation to rank which molecules to test next in the laboratory.',
    simpleExplanation: 'A formula guiding wet-lab experiments: should we test molecules predicted to be great leads (exploitation) or molecules in unexplored chemical territory where the AI is uncertain (exploration)?',
    scientificFormula: 'α(x) = E[ Utility(f(x)) | D ]',
    reference: 'Frazier, P. I. (2018). A Tutorial on Bayesian Optimization.'
  },
  exploration_exploitation: {
    term: 'Exploration vs Exploitation Trade-off',
    category: 'Active Learning',
    summary: 'The fundamental tension between refining current lead candidates vs seeking undiscovered chemical scaffolds.',
    simpleExplanation: 'Exploitation searches around known active compounds to find slight improvements. Exploration tests strange, unfamiliar chemistries that might reveal a breakthrough new drug class.',
    scientificFormula: 'UCB(x) = μ(x) + β · σ(x)   (β controls exploration strength)',
    reference: 'Srinivas et al. (2010). Gaussian Process Optimization in the Bandit Setting.'
  },
  hypervolume: {
    term: 'Hypervolume Improvement (EHVI)',
    category: 'Optimization',
    summary: 'The measure of multi-dimensional volume dominated by a Pareto front relative to an anti-ideal reference point.',
    simpleExplanation: 'Quantifies how much the entire set of discovery leads improves across all objectives at once. Higher hypervolume means better overall quality and diversity.',
    scientificFormula: 'HV(P, r) = Volume( ⋃_{x ∈ P} [f(x), r] )',
    reference: 'Emmerich et al. (2011) / Daulton et al. (2020, BoTorch qEHVI).'
  },
  gaussian_process: {
    term: 'Gaussian Process (GP) Surrogate',
    category: 'Active Learning',
    summary: 'A non-parametric Bayesian model defining a probability distribution over functions.',
    simpleExplanation: 'An intelligent AI surrogate that not only predicts property values (mean μ) for unknown molecules, but also computes exact epistemic uncertainty (variance σ^2) conditioned on real assay results.',
    scientificFormula: 'f(x) ~ GP( m(x), k(x, x\') )',
    reference: 'Rasmussen & Williams (2006). Gaussian Processes for Machine Learning.'
  }
};

interface TooltipProps {
  id: string;
  children?: React.ReactNode;
}

export const ScientificTooltip: React.FC<TooltipProps> = ({ id, children }) => {
  const [open, setOpen] = useState(false);
  const concept = SCIENTIFIC_GLOSSARY[id.toLowerCase()];

  if (!concept) {
    return <>{children}</>;
  }

  return (
    <span className="relative inline-flex items-center gap-1 group">
      {children}
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(!open);
        }}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="text-cyan-400/80 hover:text-cyan-300 transition-colors cursor-pointer inline-flex items-center p-0.5 focus:outline-none"
        title="View scientific definition"
        aria-label={`Definition of ${concept.term}`}
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {open && (
        <div 
          className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-80 sm:w-96 p-3.5 rounded-xl bg-slate-900 border border-slate-700/80 shadow-2xl backdrop-blur-xl text-left pointer-events-none animate-in fade-in zoom-in-95 duration-150"
          style={{ minWidth: '280px' }}
        >
          <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-2 mb-2">
            <div>
              <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                {concept.category}
              </span>
              <h5 className="text-xs font-bold text-white mt-1 leading-snug">
                {concept.term}
              </h5>
            </div>
            <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-1" />
          </div>

          <p className="text-[11px] text-slate-300 leading-relaxed mb-2 font-normal">
            {concept.simpleExplanation}
          </p>

          {concept.scientificFormula && (
            <div className="p-1.5 rounded bg-slate-950 border border-slate-800/80 font-mono text-[10px] text-cyan-300 mb-2 overflow-x-auto">
              {concept.scientificFormula}
            </div>
          )}

          {concept.reference && (
            <p className="text-[9px] text-slate-400 italic">
              Ref: {concept.reference}
            </p>
          )}

          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-slate-700" />
        </div>
      )}
    </span>
  );
};

export const ConceptHelpModal: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  const [filterCategory, setFilterCategory] = useState<string>('All');
  const [search, setSearch] = useState<string>('');

  if (!isOpen) return null;

  const categories = ['All', 'Structure', 'Prediction', 'Uncertainty', 'Optimization', 'Active Learning'];

  const filteredConcepts = Object.values(SCIENTIFIC_GLOSSARY).filter(item => {
    const matchCat = filterCategory === 'All' || item.category === filterCategory;
    const matchSearch = item.term.toLowerCase().includes(search.toLowerCase()) || 
                        item.simpleExplanation.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-950/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <BookOpen className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-white text-base">Scientific Glossary & Theoretical Principles</h3>
              <p className="text-xs text-slate-400">Comprehensive reference of computational chemistry and ML terminology used across MoleculeMind</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Filters */}
        <div className="p-4 border-b border-slate-800 flex flex-col sm:flex-row gap-3 bg-slate-900/60">
          <input
            type="text"
            placeholder="Search concepts, equations, properties..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
          <div className="flex flex-wrap gap-1.5">
            {categories.map(cat => (
              <button
                key={cat}
                onClick={() => setFilterCategory(cat)}
                className={`px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  filterCategory === cat
                    ? 'bg-cyan-500 text-slate-950 font-semibold shadow'
                    : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {filteredConcepts.map(concept => (
            <div key={concept.term} className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-cyan-500/30 transition-all">
              <div className="flex items-center justify-between gap-2 mb-2">
                <h4 className="text-sm font-bold text-white">{concept.term}</h4>
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  {concept.category}
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed mb-2.5">
                {concept.simpleExplanation}
              </p>
              {concept.scientificFormula && (
                <div className="p-2 rounded bg-slate-900 border border-slate-800 font-mono text-xs text-cyan-300 mb-2 overflow-x-auto">
                  {concept.scientificFormula}
                </div>
              )}
              {concept.reference && (
                <div className="text-[10px] text-slate-500 italic">
                  Academic Citation: {concept.reference}
                </div>
              )}
            </div>
          ))}
          {filteredConcepts.length === 0 && (
            <div className="text-center py-12 text-slate-500 text-xs">
              No matching scientific concepts found.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/70 text-right">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-semibold transition-colors"
          >
            Close Glossary
          </button>
        </div>
      </div>
    </div>
  );
};
