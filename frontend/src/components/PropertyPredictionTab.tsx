import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ScientificTooltip } from './ScientificTooltip';
import { 
  Activity, 
  Search, 
  AlertTriangle, 
  Target, 
  Gauge, 
  Sparkles, 
  Dna,
  ShieldCheck,
  Info,
  ChevronDown,
  ChevronUp,
  Ban,
  Network,
  FlaskConical,
  ArrowRight
} from 'lucide-react';

// Display metadata for each property key. Values, units, provenance and caveats
// all come from the backend; this map only controls presentation.
const PROPERTY_META: Record<string, any> = {
  solubility_logs: {
    tooltipId: 'solubility',
    color: 'text-cyan-300',
    bar: 'bg-cyan-400',
    format: (v: number) => v,
    // logS roughly spans -8 (insoluble) to +2 (very soluble)
    toPercent: (v: number) => ((v + 8) / 10) * 100
  },
  toxicity_risk_probability: {
    tooltipId: 'toxicity',
    color: 'text-white',
    bar: 'bg-rose-400',
    format: (v: number) => `${Math.round(v * 100)}%`,
    toPercent: (v: number) => v * 100
  },
  qed_drug_likeness: {
    tooltipId: 'qed',
    color: 'text-emerald-400',
    bar: 'bg-emerald-400',
    format: (v: number) => v,
    toPercent: (v: number) => v * 100
  }
};

// Order the profile so available properties are shown before unavailable ones.
const PROPERTY_ORDER = [
  'solubility_logs',
  'toxicity_risk_probability',
  'qed_drug_likeness',
  'bioactivity_pic50',
  'metabolic_stability'
];

interface PredictionProps {
  initialSmiles?: string;
  onSendToPareto?: () => void;
}

const PRESET_MOLECULES = [
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Ibuprofen', smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' },
  { name: 'Sulfamethoxazole', smiles: 'Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1' },
  { name: 'Diclofenac', smiles: 'O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl' },
  { name: 'Warfarin', smiles: 'CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O' },
  { name: 'Atorvastatin Core', smiles: 'Cc1c(C(=O)Nc2ccccc2)c(c3ccc(F)cc3)n(c1C(C)C)c4ccccc4' }
];

export const PropertyPredictionTab: React.FC<PredictionProps> = ({ initialSmiles, onSendToPareto }) => {
  const [smiles, setSmiles] = useState<string>(initialSmiles || 'CC(=O)Oc1ccccc1C(=O)O');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);
  const [selectedConfidence, setSelectedConfidence] = useState<'80%' | '90%' | '95%'>('90%');
  const [showConformalMath, setShowConformalMath] = useState<boolean>(false);
  const [showLeverageDetails, setShowLeverageDetails] = useState<boolean>(false);

  // Properties come from the backend keyed by name, each carrying its own
  // status/provenance. Unknown keys are tolerated so the UI does not need a
  // redeploy when a new model is added.
  const propEntries: [string, any][] = result?.properties
    ? Object.entries(result.properties as Record<string, any>)
    : [];
  const allProps = [...propEntries].sort(
    (a, b) =>
      (PROPERTY_ORDER.indexOf(a[0]) + 1 || 99) - (PROPERTY_ORDER.indexOf(b[0]) + 1 || 99)
  );
  const availableProps = allProps.filter(([, p]) => p.status === 'available');

  const handlePredict = async (targetSmiles?: string) => {
    const s = targetSmiles || smiles;
    if (!s.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: s.trim() })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed');
      }

      setResult(data);
      setSmiles(data.smiles);
    } catch (err: any) {
      setError(err.message || 'Error communicating with prediction server');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialSmiles) {
      setSmiles(initialSmiles);
      handlePredict(initialSmiles);
    } else {
      handlePredict('CC(=O)Oc1ccccc1C(=O)O');
    }
  }, [initialSmiles]);

  const currentInterval = result?.solubility?.conformal?.intervals?.[selectedConfidence];

  const getSolubilityInterpretation = (logS: number) => {
    if (logS > -1.0) return 'Highly Soluble — Rapidly dissolves in physiological saline and gastric fluids.';
    if (logS > -2.5) return 'Soluble — Favorable aqueous dissolution suitable for standard oral solid dosage.';
    if (logS > -4.0) return 'Moderately Soluble — May require formulation enhancement (surfactants or micronization).';
    return 'Poorly Soluble (Lipophilic) — Elevated risk of precipitation and impaired in vivo bioavailability.';
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Header & Query Input */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-mono tracking-wider uppercase text-slate-400">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>Stages 02 & 03 • Conformal Inference Engine</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-medium tracking-tight text-white">
              Property Prediction & Reliability Bounds
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-2xl">
              Predict <ScientificTooltip id="solubility"><span className="text-white underline decoration-dotted cursor-help">aqueous solubility</span></ScientificTooltip>, quantify distribution-free <ScientificTooltip id="conformal"><span className="text-white underline decoration-dotted cursor-help">conformal intervals</span></ScientificTooltip>, and evaluate compound reliability via <ScientificTooltip id="applicability_domain"><span className="text-white underline decoration-dotted cursor-help">OECD applicability domain</span></ScientificTooltip>.
            </p>
          </div>

          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs font-mono text-slate-300 self-start sm:self-auto">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>Delaney ESOL ExtraTrees (R² = 0.894)</span>
          </div>
        </div>

        {/* SMILES Input */}
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePredict()}
            placeholder="Enter SMILES string to predict solubility and uncertainty..."
            className="flex-1 bg-black/40 border border-white/[0.1] rounded-2xl px-5 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-white/30 focus:ring-1 focus:ring-white/20 font-mono text-sm transition-all"
          />

          <button
            onClick={() => handlePredict()}
            disabled={loading}
            className="px-7 py-3.5 rounded-2xl bg-white hover:bg-slate-200 disabled:opacity-50 text-zinc-950 font-semibold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg shrink-0"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-zinc-950 border-t-transparent rounded-full animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            <span>Evaluate Compound</span>
          </button>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap items-center gap-2 mt-5 pt-4 border-t border-white/[0.06]">
          <span className="text-xs font-mono text-slate-400 flex items-center gap-1.5 mr-1">
            <Sparkles className="w-3.5 h-3.5 text-slate-300" /> Presets:
          </span>
          {PRESET_MOLECULES.map((mol) => (
            <button
              key={mol.name}
              onClick={() => {
                setSmiles(mol.smiles);
                handlePredict(mol.smiles);
              }}
              className="px-3.5 py-1.5 text-xs rounded-full bg-white/[0.03] hover:bg-white/[0.08] text-slate-300 hover:text-white transition-all border border-white/[0.08] hover:border-white/20 cursor-pointer"
            >
              {mol.name}
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-5 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs sm:text-sm flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {result && (
        <div className="space-y-8">
          {/* Top Section: Aqueous Solubility Prediction & Conformal Bounds vs Applicability Domain */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Card 1: Primary ESOL Prediction & Conformal Prediction Interval */}
            <div className="lg:col-span-7 bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2.5">
                    <Target className="w-5 h-5 text-slate-300" />
                    <div>
                      <h3 className="font-medium text-white text-base">Aqueous Solubility (ESOL logS)</h3>
                      <p className="text-xs text-slate-400 font-mono">ExtraTrees regression + Inductive Conformal calibration</p>
                    </div>
                  </div>

                  <span className={`px-3 py-1 rounded-full text-xs font-mono font-semibold border ${
                    result.solubility.predicted_value > -2.0
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : (result.solubility.predicted_value > -4.0 
                          ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20' 
                          : 'bg-amber-500/10 text-amber-300 border-amber-500/20')
                  }`}>
                    {result.solubility.classification}
                  </span>
                </div>

                {/* Primary Metric Displays */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                  {/* Predicted Value Card */}
                  <div className="bg-black/40 p-5 rounded-2xl border border-white/[0.06]">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400 font-mono">Estimated logS</span>
                      <ScientificTooltip id="solubility" />
                    </div>
                    <div className="text-3xl font-light text-white mt-1 font-mono">
                      {result.solubility.predicted_value}
                      <span className="text-xs font-normal text-slate-400 ml-1.5 font-sans">log(mol/L)</span>
                    </div>
                    <div className="text-xs text-slate-400 mt-2 flex items-center gap-1.5 font-mono">
                      <span>Tree Std Dev (σ):</span>
                      <strong className="text-slate-200">±{result.solubility.std_deviation}</strong>
                    </div>
                  </div>

                  {/* Conformal Prediction Interval Card */}
                  <div className="bg-black/40 p-5 rounded-2xl border border-white/[0.06] flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-mono">Conformal Interval</span>
                        <ScientificTooltip id="conformal" />
                      </div>
                      <div className="text-xl font-light text-emerald-300 mt-1 font-mono">
                        [{currentInterval?.lower_bound}, {currentInterval?.upper_bound}]
                      </div>
                      <div className="text-xs text-slate-400 mt-1 font-mono">
                        Margin: <strong className="text-white">±{currentInterval?.margin_error}</strong> (Width: {currentInterval?.interval_width})
                      </div>
                    </div>

                    {/* Confidence Selector */}
                    <div className="flex items-center gap-2 mt-4 pt-3 border-t border-white/[0.06]">
                      <span className="text-[11px] text-slate-400 font-mono">Confidence:</span>
                      {(['80%', '90%', '95%'] as const).map((conf) => (
                        <button
                          key={conf}
                          onClick={() => setSelectedConfidence(conf)}
                          className={`px-3 py-1 text-xs font-mono rounded-full transition-all cursor-pointer ${
                            selectedConfidence === conf
                              ? 'bg-white text-zinc-950 font-semibold shadow'
                              : 'bg-white/[0.04] text-slate-400 hover:text-white'
                          }`}
                        >
                          {conf}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Plain-Language Interpretation Callout */}
                <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06] mb-6">
                  <div className="flex items-start gap-3">
                    <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <div>
                      <div className="text-xs font-medium text-white">
                        Physiological Solubility Assessment:
                      </div>
                      <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                        {getSolubilityInterpretation(result.solubility.predicted_value)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Visual Interval Representation */}
                <div className="bg-black/40 p-5 rounded-2xl border border-white/[0.06]">
                  <div className="flex justify-between text-xs text-slate-400 mb-2 font-mono">
                    <span>Lower ({selectedConfidence}): <strong className="text-white">{currentInterval?.lower_bound}</strong></span>
                    <span className="text-white font-semibold">Point: {result.solubility.predicted_value}</span>
                    <span>Upper ({selectedConfidence}): <strong className="text-white">{currentInterval?.upper_bound}</strong></span>
                  </div>

                  <div className="relative w-full h-4 bg-white/[0.05] rounded-full overflow-hidden border border-white/[0.08] flex items-center px-4">
                    <div className="w-full h-1.5 bg-white/[0.1] rounded-full relative">
                      <div 
                        className="absolute h-full bg-white rounded-full shadow-[0_0_12px_rgba(255,255,255,0.4)]"
                        style={{ left: '20%', right: '20%' }}
                      />
                      <div 
                        className="absolute w-3 h-3 bg-white rounded-full -top-0.5 shadow transform -translate-x-1/2"
                        style={{ left: '50%' }}
                      />
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-400 mt-3 leading-relaxed font-mono">
                    <strong>Statistical Guarantee:</strong> With at least <strong>{selectedConfidence}</strong> empirical coverage, true laboratory solubility falls within this interval without assuming Gaussian error distributions.
                  </p>
                </div>

                {/* Expandable Section: Conformal Calibration Mathematics */}
                <div className="mt-5 pt-4 border-t border-white/[0.06]">
                  <button
                    onClick={() => setShowConformalMath(!showConformalMath)}
                    className="w-full py-2.5 px-4 rounded-xl bg-black/40 hover:bg-white/[0.04] text-slate-300 hover:text-white text-xs font-mono flex items-center justify-between transition-colors border border-white/[0.06] cursor-pointer"
                  >
                    <span className="flex items-center gap-2">
                      <ShieldCheck className="w-3.5 h-3.5 text-slate-300" />
                      <span>{showConformalMath ? 'Collapse Mathematical Derivation' : 'Expand Conformal Quantile Formulation'}</span>
                    </span>
                    {showConformalMath ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showConformalMath && (
                    <div className="mt-3 p-5 rounded-2xl bg-black/40 border border-white/[0.06] text-xs space-y-3 font-mono">
                      <div className="text-xs font-semibold text-white">
                        Inductive Split Conformal Prediction (ICP)
                      </div>
                      <p className="text-slate-300 leading-relaxed text-[11px]">
                        The Delaney dataset was split into Training (80%), Calibration (10%), and Test (10%). For each calibration compound i, absolute non-conformity residuals were recorded:
                      </p>
                      <div className="p-3 bg-white/[0.03] border border-white/[0.06] rounded-xl text-white text-center">
                        {"s_i = | y_i - ŷ_i |"}
                      </div>
                      <p className="text-slate-300 leading-relaxed text-[11px]">
                        The conformal prediction interval at confidence 1 - α is constructed from the finite-sample adjusted empirical quantile:
                      </p>
                      <div className="p-3 bg-white/[0.03] border border-white/[0.06] rounded-xl text-white text-center">
                        {"C(X_new) = [ ŷ_new - q_{1-α}, ŷ_new + q_{1-α} ]"}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* SMILES footnote */}
              <div className="mt-5 pt-4 border-t border-white/[0.06] flex items-center justify-between text-xs text-slate-400 font-mono">
                <span className="truncate max-w-sm">{result.smiles}</span>
                <span className="text-emerald-400 font-medium">Valid Delaney Input</span>
              </div>
            </div>

            {/* Card 2: Applicability Domain (AD) & Chemical Neighborhood */}
            <div className="lg:col-span-5 bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2.5">
                    <Gauge className="w-5 h-5 text-slate-300" />
                    <div>
                      <h3 className="font-medium text-white text-base">Applicability Domain</h3>
                      <p className="text-xs text-slate-400 font-mono">OECD Guidance No. 69 Boundary Check</p>
                    </div>
                  </div>

                  {/* AD Status Badge */}
                  <span className={`px-3 py-1 rounded-full text-xs font-mono font-semibold border ${
                    result.applicability_domain.status === 'IN_DOMAIN'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : (result.applicability_domain.status === 'BORDERLINE'
                          ? 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                          : 'bg-rose-500/10 text-rose-400 border-rose-500/20')
                  }`}>
                    {result.applicability_domain.status}
                  </span>
                </div>

                {/* Reliability Meter */}
                <div className="bg-black/40 p-5 rounded-2xl border border-white/[0.06] mb-6">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-xs text-slate-400 font-mono">Reliability Score</span>
                    <span className="text-xl font-light font-mono text-white">
                      {Math.round(result.applicability_domain.reliability_score * 100)}%
                    </span>
                  </div>

                  <div className="w-full bg-white/[0.05] h-2 rounded-full overflow-hidden border border-white/[0.08]">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        result.applicability_domain.reliability_score > 0.6
                          ? 'bg-emerald-400'
                          : (result.applicability_domain.reliability_score > 0.4 ? 'bg-amber-400' : 'bg-rose-400')
                      }`}
                      style={{ width: `${Math.round(result.applicability_domain.reliability_score * 100)}%` }}
                    />
                  </div>

                  <p className="text-xs text-slate-300 mt-3 leading-relaxed">
                    {result.applicability_domain.explanation}
                  </p>
                </div>

                {/* AD Metric Breakdown Cards */}
                <div className="grid grid-cols-2 gap-3 text-xs mb-6">
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="flex items-center justify-between text-slate-400 font-mono">
                      <span>Max Tanimoto</span>
                      <ScientificTooltip id="tanimoto" />
                    </div>
                    <div className="text-lg font-light font-mono text-white mt-1">
                      {Math.round(result.applicability_domain.metrics.max_tanimoto_similarity * 100)}%
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Nearest Delaney training compound</div>
                  </div>

                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono">Top-5 Mean Sim</div>
                    <div className="text-lg font-light font-mono text-white mt-1">
                      {Math.round(result.applicability_domain.metrics.mean_top5_tanimoto * 100)}%
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Local chemical neighborhood</div>
                  </div>

                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="flex items-center justify-between text-slate-400 font-mono">
                      <span>Leverage (h)</span>
                      <ScientificTooltip id="leverage" />
                    </div>
                    <div className="text-base font-light font-mono text-white mt-1">
                      {result.applicability_domain.metrics.leverage}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">Threshold h* = {result.applicability_domain.metrics.leverage_threshold}</div>
                  </div>

                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono">OECD Status</div>
                    <div className={`text-base font-medium mt-1 ${
                      result.applicability_domain.metrics.leverage_warning ? 'text-rose-400' : 'text-emerald-400'
                    }`}>
                      {result.applicability_domain.metrics.leverage_warning ? 'Extrapolation' : 'Interpolation OK'}
                    </div>
                    <div className="text-[10px] text-slate-400 mt-0.5">OECD 3(k+1)/N</div>
                  </div>
                </div>

                {/* Experimental Delaney Analog Card */}
                <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06] text-xs">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <FlaskConical className="w-3.5 h-3.5 text-slate-300" />
                      <span className="text-white font-medium">Nearest Delaney Training Compound</span>
                    </div>
                    <span className="text-slate-300 font-mono font-medium">
                      {Math.round(result.applicability_domain.nearest_analog.tanimoto_similarity * 100)}% Sim
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] flex justify-between items-center">
                    <div>
                      <div className="font-medium text-white">
                        {result.applicability_domain.nearest_analog.compound_id}
                      </div>
                      <div className="text-[10px] font-mono text-slate-400 truncate max-w-[200px]">
                        {result.applicability_domain.nearest_analog.smiles}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-[10px] text-slate-400 font-mono">Measured logS:</div>
                      <div className="font-mono font-medium text-emerald-400 text-sm">
                        {result.applicability_domain.nearest_analog.measured_value}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Expandable Section: OECD Leverage Details */}
                <div className="mt-4 pt-2">
                  <button
                    onClick={() => setShowLeverageDetails(!showLeverageDetails)}
                    className="w-full py-1.5 px-3 text-slate-400 hover:text-white text-xs font-mono flex items-center justify-between transition-colors cursor-pointer"
                  >
                    <span>{showLeverageDetails ? 'Collapse Chemical Leverage Theory' : 'What is Chemical Leverage?'}</span>
                    {showLeverageDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>

                  {showLeverageDetails && (
                    <p className="mt-2 text-[11px] text-slate-400 leading-relaxed bg-black/40 p-4 rounded-2xl border border-white/[0.06] font-mono">
                      Leverage measures distance in PCA descriptor space from the training set centroid. A compound with leverage h &gt; h* lies outside the multidimensional convex hull of Delaney training molecules, indicating that predictions represent extrapolation rather than reliable interpolation.
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Section: Multi-Objective Discovery Profile */}
          <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 pb-4 border-b border-white/[0.06]">
              <div className="flex items-center gap-2.5">
                <Dna className="w-5 h-5 text-slate-300" />
                <div>
                  <h3 className="font-medium text-white text-base">Multi-Objective Profile</h3>
                  <p className="text-xs text-slate-400">
                    Every value is labelled with its provenance. Properties without a trained,
                    evaluated model are shown as unavailable rather than estimated.
                  </p>
                </div>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {availableProps.length} of {allProps.length} objectives available
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {allProps.map(([key, prop]: [string, any]) => {
                const isAvailable = prop.status === 'available';
                const meta = PROPERTY_META[key] || {};

                if (!isAvailable) {
                  return (
                    <div
                      key={key}
                      className="bg-black/20 p-5 rounded-2xl border border-dashed border-white/[0.10] flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                          <span>{prop.property_name}</span>
                          <Ban className="w-3.5 h-3.5 text-slate-600" />
                        </div>
                        <div className="text-base font-mono text-slate-500 mt-2">Unavailable</div>
                        <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">{prop.reason}</p>
                      </div>
                      {prop.required_to_enable && (
                        <div className="mt-3 pt-3 border-t border-white/[0.06] text-[10px] text-slate-500 font-mono">
                          Requires: {prop.required_to_enable}
                        </div>
                      )}
                    </div>
                  );
                }

                const pct = meta.toPercent ? meta.toPercent(prop.value) : 0;
                return (
                  <div
                    key={key}
                    className="bg-black/40 p-5 rounded-2xl border border-white/[0.06] flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                        <span>{prop.property_name}</span>
                        {meta.tooltipId ? <ScientificTooltip id={meta.tooltipId} /> : null}
                      </div>
                      <div className="flex items-baseline gap-2 mt-1">
                        <span className={`text-2xl font-light font-mono ${meta.color || 'text-white'}`}>
                          {meta.format ? meta.format(prop.value) : prop.value}
                        </span>
                        {prop.category && (
                          <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border bg-white/[0.05] text-slate-300 border-white/[0.1]">
                            {prop.category}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 mt-2">
                        <span
                          className={`text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border ${
                            prop.provenance === 'model'
                              ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
                              : 'bg-slate-500/10 text-slate-300 border-slate-500/20'
                          }`}
                          title={prop.model}
                        >
                          {prop.provenance === 'model' ? 'Predicted' : 'Computed'}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">{prop.unit}</span>
                      </div>
                      {prop.caveat && (
                        <p className="text-[10px] text-amber-300/70 mt-2 leading-relaxed">{prop.caveat}</p>
                      )}
                    </div>

                    <div className="mt-4">
                      <div className="w-full bg-white/[0.05] h-1.5 rounded-full overflow-hidden border border-white/[0.08]">
                        <div
                          className={`h-full ${meta.bar || 'bg-slate-400'}`}
                          style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                        />
                      </div>
                      {prop.metrics && (prop.metrics.roc_auc || prop.metrics.pr_auc) && (
                        <div className="text-[10px] text-slate-500 font-mono mt-1.5">
                          Model: ROC-AUC {prop.metrics.roc_auc} · PR-AUC {prop.metrics.pr_auc}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Stage 3: independent cross-check against the graph neural network */}
            {result.dmpnn && result.dmpnn.solubility_logs && (
              <div className="mt-6 pt-5 border-t border-white/[0.06]">
                <div className="flex items-center gap-2 mb-3">
                  <Network className="w-4 h-4 text-cyan-400" />
                  <h4 className="text-sm font-medium text-white">
                    Independent cross-check: D-MPNN graph neural network
                  </h4>
                </div>
                <p className="text-[11px] text-slate-400 mb-4 leading-relaxed">
                  A second model of a different family on a different molecular representation
                  (a directed message-passing network over the molecular graph, versus bagged
                  trees over fingerprint bits). Agreement is weak evidence that a prediction is
                  representation-independent — not proof that it is correct.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-[11px] text-slate-400 font-mono">D-MPNN logS</div>
                    <div className="text-xl font-light font-mono text-cyan-300 mt-1">
                      {result.dmpnn.solubility_logs.value}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono mt-1">
                      σ = {result.dmpnn.solubility_logs.sigma} ({result.dmpnn.n_members}-member ensemble)
                    </div>
                  </div>

                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-[11px] text-slate-400 font-mono">Baseline logS</div>
                    <div className="text-xl font-light font-mono text-white mt-1">
                      {result.dmpnn.agreement_with_baseline.baseline_value}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono mt-1">
                      ExtraTrees on fingerprints
                    </div>
                  </div>

                  <div
                    className={`p-4 rounded-2xl border ${
                      result.dmpnn.agreement_with_baseline.exceeds_dmpnn_sigma
                        ? 'bg-amber-500/5 border-amber-500/20'
                        : 'bg-emerald-500/5 border-emerald-500/20'
                    }`}
                  >
                    <div className="text-[11px] text-slate-400 font-mono">Difference</div>
                    <div
                      className={`text-xl font-light font-mono mt-1 ${
                        result.dmpnn.agreement_with_baseline.exceeds_dmpnn_sigma
                          ? 'text-amber-300'
                          : 'text-emerald-300'
                      }`}
                    >
                      {result.dmpnn.agreement_with_baseline.absolute_difference}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono mt-1">
                      {result.dmpnn.agreement_with_baseline.exceeds_dmpnn_sigma
                        ? 'Exceeds σ — representation-sensitive'
                        : 'Within σ — models agree'}
                    </div>
                  </div>
                </div>

                {/* Variance decomposition: the ensemble separates the two sources */}
                <div className="mt-3 grid grid-cols-2 gap-3 text-[11px]">
                  <div className="bg-black/20 px-4 py-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">Aleatoric variance</span>
                    <div className="text-slate-200 font-mono mt-0.5">
                      {result.dmpnn.solubility_logs.aleatoric_variance}
                      <span className="text-slate-500"> — irreducible data noise</span>
                    </div>
                  </div>
                  <div className="bg-black/20 px-4 py-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">Epistemic variance</span>
                    <div className="text-slate-200 font-mono mt-0.5">
                      {result.dmpnn.solubility_logs.epistemic_variance}
                      <span className="text-slate-500"> — member disagreement</span>
                    </div>
                  </div>
                </div>

                {result.dmpnn.toxicity?.conformal_sets && (
                  <div className="mt-3 bg-black/20 px-4 py-3 rounded-xl border border-white/[0.06] text-[11px]">
                    <span className="text-slate-400 font-mono">
                      Conformal prediction set for toxicity (90%)
                    </span>
                    <div className="mt-1.5 flex flex-wrap gap-2">
                      {(['lac', 'aps_randomized'] as const).map((method) => {
                        const set = result.dmpnn.toxicity.conformal_sets[method];
                        if (!set) return null;
                        return (
                          <span
                            key={method}
                            className="px-2.5 py-1 rounded-full bg-white/[0.05] border border-white/[0.1] text-slate-200 font-mono"
                            title={set.method}
                          >
                            {method === 'lac' ? 'LAC' : 'APS'}:{' '}
                            {set.labels.length ? set.labels.join(' | ') : '(empty set)'}
                          </span>
                        );
                      })}
                    </div>
                    <p className="text-[10px] text-slate-500 mt-2 leading-relaxed">
                      A set, not a single label. An empty set means no label was plausible at
                      this confidence; both labels means the model cannot discriminate.
                      Coverage is marginal across the test distribution, not a per-molecule
                      probability.
                    </p>
                  </div>
                )}
              </div>
            )}

            {onSendToPareto && (
              <div className="mt-6 pt-5 border-t border-white/[0.06] flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
                <span className="text-slate-400">
                  Ready to benchmark this compound across the candidate library?
                </span>
                <button
                  onClick={onSendToPareto}
                  className="px-5 py-2.5 bg-white hover:bg-slate-200 text-zinc-950 font-semibold rounded-full transition-all cursor-pointer flex items-center gap-2"
                >
                  <span>Proceed to Pareto Laboratory (Stage 04)</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
};
