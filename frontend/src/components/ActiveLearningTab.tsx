import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import { motion } from 'framer-motion';
import { ScientificTooltip } from './ScientificTooltip';
import { 
  Cpu, 
  FlaskConical, 
  TrendingUp, 
  RefreshCw, 
  CheckSquare, 
  Square, 
  Zap,
  Info,
  Sparkles
} from 'lucide-react';

interface ActiveLearningProps {
  onSelectCandidate?: (smiles: string) => void;
}

export const ActiveLearningTab: React.FC<ActiveLearningProps> = ({ onSelectCandidate }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [data, setData] = useState<any | null>(null);
  const [strategy, setStrategy] = useState<string>('ucb');
  const [beta, setBeta] = useState<number>(2.0);
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);
  const [showWorkflowHelp, setShowWorkflowHelp] = useState<boolean>(false);

  const fetchPool = async (strat?: string, b?: number) => {
    const s = strat || strategy;
    const currentBeta = b !== undefined ? b : beta;
    setLoading(true);
    try {
      const res = await fetch(`/api/active-learning/pool?strategy=${s}&beta=${currentBeta}`);
      const json = await res.json();
      setData(json);
      
      // Auto-select the top recommendation if no selection
      const unassayed = (json.candidates || []).filter((c: any) => !c.assayed);
      if (unassayed.length > 0 && selectedCandidates.length === 0) {
        const top = [...unassayed].sort((a, b) => (b.acquisition_score || 0) - (a.acquisition_score || 0))[0];
        if (top) {
          setSelectedCandidates([top.id]);
        }
      }
    } catch (e) {
      console.error("Failed to load active learning pool:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPool();
  }, []);

  const handleStrategyChange = (newStrat: string) => {
    setStrategy(newStrat);
    fetchPool(newStrat, beta);
  };

  const handleBetaChange = (newBeta: number) => {
    setBeta(newBeta);
    fetchPool(strategy, newBeta);
  };

  const toggleSelect = (id: string) => {
    setSelectedCandidates(prev => 
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  const runAssay = async () => {
    if (selectedCandidates.length === 0) return;
    setActionLoading(true);
    try {
      const res = await fetch('/api/active-learning/assay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_ids: selectedCandidates })
      });
      await res.json();
      confetti({
        particleCount: 70,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#ffffff', '#38bdf8', '#34d399', '#a855f7']
      });
      setSelectedCandidates([]);
      await fetchPool();
    } catch (e) {
      console.error("Failed to run assay:", e);
    } finally {
      setActionLoading(false);
    }
  };

  const resetCampaign = async () => {
    if (!window.confirm("Reset Active Learning campaign to Round 0?")) return;
    setActionLoading(true);
    try {
      await fetch('/api/active-learning/reset', { method: 'POST' });
      setSelectedCandidates([]);
      await fetchPool();
    } catch (e) {
      console.error("Failed to reset campaign:", e);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center h-96 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] text-slate-400">
        <span className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-sm font-mono">Conditioning BoTorch SingleTaskGP Gaussian Process...</p>
      </div>
    );
  }

  const candidates = data?.candidates || [];
  const unassayed = candidates.filter((c: any) => !c.assayed);
  const assayed = candidates.filter((c: any) => c.assayed);

  const rankedUnassayed = [...unassayed].sort((a: any, b: any) => (b.acquisition_score || 0) - (a.acquisition_score || 0));

  // Only measurements released by an executed assay are available here; the
  // hidden oracle is held server-side and is never sent to the client.
  const bestDiscovered = assayed.length > 0
    ? [...assayed].sort((a: any, b: any) => (b.measured_value ?? -Infinity) - (a.measured_value ?? -Infinity))[0]
    : null;

  const strategyExplanations: Record<string, { title: string; simple: string; math: string }> = {
    ucb: {
      title: 'Upper Confidence Bound (qUCB)',
      simple: 'Balances high predicted potency with high epistemic uncertainty. The parameter β controls the exploration appetite.',
      math: 'α_UCB(x) = μ(x) + β · σ(x)'
    },
    ei: {
      title: 'Expected Improvement (EI)',
      simple: 'Quantifies the mathematical probability and magnitude of surpassing the current best discovered compound in the library.',
      math: 'α_EI(x) = E[ max(0, f(x) - f*) ]'
    },
    uncertainty: {
      title: 'Uncertainty Sampling (Pure Exploration)',
      simple: 'Prioritizes candidate molecules in chemical regions where the surrogate GP is most uncertain (maximum variance σ).',
      math: 'α_Uncertainty(x) = σ(x)'
    },
    pareto_hv: {
      title: 'Hypervolume Improvement (Multi-Objective)',
      simple: 'Measures how much each molecule expands the dominated multi-dimensional space across both potency and solubility.',
      math: 'α_HV(x) = Volume(Front ∪ {f(x)}) - Volume(Front)'
    }
  };

  const currentStratInfo = strategyExplanations[strategy] || strategyExplanations.ucb;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Educational Header */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="flex items-center gap-2 text-xs font-mono tracking-wider uppercase text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-purple-400" />
              <span>Stage 05 • Bayesian Experimental Acquisition</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-medium tracking-tight text-white">
              Active Learning: BoTorch Acquisition Loop
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-normal">
              Active learning replaces unguided high-throughput screening. A <ScientificTooltip id="gaussian_process"><span className="text-white underline decoration-dotted cursor-help">Gaussian Process (GP) surrogate</span></ScientificTooltip> predicts both biological utility (mean μ) and uncertainty (variance σ). The <ScientificTooltip id="acquisition"><span className="text-white underline decoration-dotted cursor-help">acquisition policy</span></ScientificTooltip> prioritizes promising compounds for wet-lab validation.
            </p>
          </div>

          <div className="flex flex-wrap lg:flex-col gap-2 shrink-0">
            <button
              onClick={() => setShowWorkflowHelp(!showWorkflowHelp)}
              className="px-4 py-2 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.1] text-xs font-mono text-slate-300 flex items-center gap-2 transition-colors cursor-pointer"
            >
              <Info className="w-3.5 h-3.5 text-slate-300" />
              <span>{showWorkflowHelp ? 'Hide Discovery Flow' : 'Explain Discovery Flow'}</span>
            </button>
          </div>
        </div>

        {/* 6-Step Workflow Breakdown */}
        {showWorkflowHelp && (
          <div className="mt-6 pt-5 border-t border-white/[0.06] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">1. Candidate Pool Partition</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">Library compounds are split into already-evaluated seeds and unassayed candidates.</p>
            </div>
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">2. Mean & Epistemic Uncertainty</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">Surrogate GP computes mean affinity (μ) and epistemic uncertainty (σ) for every structure.</p>
            </div>
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">3. Select Acquisition Strategy</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">Choose between Upper Confidence Bound (qUCB), Expected Improvement (EI), or Uncertainty sampling.</p>
            </div>
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">4. Select Candidates for Assay</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">Check the boxes of top-ranked compounds or select custom candidates for simulated assay.</p>
            </div>
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">5. Run Simulated Experiment</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">Execute "Run Wet-Lab Assay" to uncover ground-truth experimental wet-lab measurements.</p>
            </div>
            <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06]">
              <div className="font-medium text-white mb-1">6. Posterior GP Refinement</div>
              <p className="text-slate-400 text-[11px] leading-relaxed">The GP conditions on new measurements, updating its posterior and shrinking uncertainty nearby.</p>
            </div>
          </div>
        )}
      </div>

      {/* Campaign Status Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] p-6 rounded-3xl backdrop-blur-2xl">
          <div className="text-xs text-slate-400 font-mono">Iteration</div>
          <div className="text-3xl font-light font-mono text-white mt-1">Round {data?.iteration ?? 0}</div>
          <div className="text-[11px] text-slate-400 mt-1 font-mono">Sequential Bayesian loop</div>
        </div>

        <div className="bg-[#0c0d14]/70 border border-white/[0.08] p-6 rounded-3xl backdrop-blur-2xl">
          <div className="text-xs text-slate-400 font-mono">Assay Progress</div>
          <div className="text-3xl font-light font-mono text-white mt-1">
            {assayed.length} <span className="text-sm font-normal text-slate-400 font-sans">/ {candidates.length}</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1 font-mono">{unassayed.length} unassayed remaining</div>
        </div>

        <div className="bg-[#0c0d14]/70 border border-white/[0.08] p-6 rounded-3xl backdrop-blur-2xl">
          <div className="text-xs text-slate-400 font-mono">Current Lead Compound</div>
          <div className="text-base font-medium text-emerald-400 mt-1 truncate">
            {bestDiscovered ? bestDiscovered.name : 'None yet'}
          </div>
          <div className="text-[11px] text-slate-400 mt-1 font-mono">
            {bestDiscovered
              ? <>Measured logS: <strong className="text-white">{bestDiscovered.measured_value}</strong> (round {bestDiscovered.assay_round})</>
              : 'Run an assay to reveal a measurement'}
          </div>
        </div>

        <div className="bg-[#0c0d14]/70 border border-white/[0.08] p-6 rounded-3xl backdrop-blur-2xl flex flex-col justify-between">
          <div className="text-xs text-slate-400 font-mono">Campaign Controls</div>
          <div className="flex gap-2 mt-2">
            <button
              onClick={resetCampaign}
              disabled={actionLoading}
              className="px-3.5 py-1.5 bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 rounded-full text-xs font-mono flex items-center gap-1.5 transition-all cursor-pointer border border-white/[0.08]"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Reset</span>
            </button>
            <button
              onClick={() => fetchPool()}
              className="px-3.5 py-1.5 bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 rounded-full text-xs font-mono flex items-center gap-1.5 transition-all cursor-pointer border border-white/[0.08]"
            >
              <span>Sync GP</span>
            </button>
          </div>
        </div>
      </div>

      {/* BoTorch Optimization Strategy Control Panel */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/[0.06]">
          <div>
            <h3 className="font-medium text-white text-base flex items-center gap-2">
              <Cpu className="w-5 h-5 text-slate-300" />
              Acquisition Strategy Selection
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Select mathematical policy guiding candidate prioritization for the next simulated assay.
            </p>
          </div>

          {/* Strategy Pills */}
          <div className="flex flex-wrap items-center gap-1 bg-black/40 p-1 rounded-full border border-white/[0.08]">
            <button
              onClick={() => handleStrategyChange('ucb')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
                strategy === 'ucb' 
                  ? 'bg-white text-zinc-950 font-semibold shadow' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              qUCB
            </button>
            <button
              onClick={() => handleStrategyChange('ei')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
                strategy === 'ei' 
                  ? 'bg-white text-zinc-950 font-semibold shadow' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Expected Improvement
            </button>
            <button
              onClick={() => handleStrategyChange('uncertainty')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
                strategy === 'uncertainty' 
                  ? 'bg-white text-zinc-950 font-semibold shadow' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Uncertainty (Max σ)
            </button>
            <button
              onClick={() => handleStrategyChange('pareto_hv')}
              className={`px-3.5 py-1.5 rounded-full text-xs font-mono transition-all cursor-pointer ${
                strategy === 'pareto_hv' 
                  ? 'bg-white text-zinc-950 font-semibold shadow' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Hypervolume (qEHVI)
            </button>
          </div>
        </div>

        {/* Selected Strategy Box */}
        <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06] mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
            <span className="text-xs font-medium text-white flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-slate-300" />
              {currentStratInfo.title}
            </span>
            <code className="text-slate-300 font-mono text-xs bg-white/[0.04] px-3 py-1 rounded-lg border border-white/[0.06]">
              {currentStratInfo.math}
            </code>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            {currentStratInfo.simple}
          </p>
        </div>

        {/* UCB Beta Slider */}
        {strategy === 'ucb' && (
          <div className="p-4 bg-black/40 rounded-2xl border border-white/[0.06] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs font-mono">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2 font-medium text-white">
                <span>Exploration Weight (β):</span>
                <span className="text-white">{beta.toFixed(1)}</span>
                <ScientificTooltip id="exploration_exploitation" />
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                Low β (0.5 - 1.0) = <strong>Exploitation</strong>; High β (3.0 - 5.0) = <strong>Exploration</strong>.
              </p>
            </div>

            <div className="flex items-center gap-3 w-full sm:w-64">
              <span className="text-[10px] text-slate-400 font-mono">0.5</span>
              <input
                type="range"
                min="0.5"
                max="5.0"
                step="0.5"
                value={beta}
                onChange={(e) => handleBetaChange(parseFloat(e.target.value))}
                className="flex-1 accent-white cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 font-mono">5.0</span>
            </div>
          </div>
        )}
      </div>

      {/* Experiment Queue & Execution Action Bar */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2.5">
            <FlaskConical className="w-5 h-5 text-slate-300" />
            <h4 className="font-medium text-white text-base">Wet-Lab Experiment Queue</h4>
          </div>
          <p className="text-xs text-slate-300 mt-1">
            <strong>{selectedCandidates.length}</strong> candidate molecule(s) queued for evaluation in Round {data?.iteration ? data.iteration + 1 : 1}.
          </p>
          <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
            Running the assay reveals laboratory measurements, recalculates posterior covariances, and updates rankings.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <button
            onClick={() => setSelectedCandidates([])}
            className="px-4 py-2 text-xs font-mono text-slate-400 hover:text-white transition-all cursor-pointer"
          >
            Clear Selection
          </button>

          <button
            onClick={runAssay}
            disabled={actionLoading || selectedCandidates.length === 0}
            className="px-6 py-3 bg-white hover:bg-slate-200 disabled:opacity-30 text-zinc-950 font-semibold text-xs rounded-full shadow-lg transition-all flex items-center gap-2 cursor-pointer"
          >
            {actionLoading ? (
              <span className="w-4 h-4 border-2 border-zinc-950 border-t-transparent rounded-full animate-spin" />
            ) : (
              <Zap className="w-4 h-4 fill-current" />
            )}
            <span>Run Wet-Lab Assay ({selectedCandidates.length})</span>
          </button>
        </div>
      </div>

      {/* Discovery Trajectory History */}
      {data?.history && data.history.length > 1 && (
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 backdrop-blur-2xl">
          <div className="text-xs font-medium text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-slate-300" />
            <span>Discovery Trajectory Across Iterations</span>
          </div>
          <div className="flex items-center gap-3 overflow-x-auto pb-2 text-xs font-mono">
            {data.history.map((h: any, i: number) => (
              <div key={i} className="bg-black/40 p-4 rounded-2xl border border-white/[0.06] shrink-0 min-w-[200px]">
                <div className="text-[10px] text-slate-400 uppercase tracking-wider">Round {h.round}</div>
                <div className="text-white font-medium text-base mt-1">Best logS: {h.best_revealed_logs ?? '—'}</div>
                <div className="text-xs text-slate-300 truncate mt-1">{h.best_compound}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Candidate Recommendation Table */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-6 pb-4 border-b border-white/[0.06]">
          <div>
            <h4 className="font-medium text-white text-base">BoTorch Acquisition Rankings</h4>
            <p className="text-xs text-slate-400 mt-0.5">
              Unassayed candidates prioritized by surrogate acquisition policy.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 bg-black/40 px-3 py-1 rounded-full border border-white/[0.06]">
            {unassayed.length} candidates remaining
          </span>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-white/[0.06]">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-black/50 text-slate-400 font-medium uppercase tracking-wider text-[11px] font-mono border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-3 w-12 text-center">Queue</th>
                <th className="py-3 px-4">Compound</th>
                <th className="py-3 px-4">Acquisition Score</th>
                <th className="py-3 px-4">Predicted Mean (μ)</th>
                <th className="py-3 px-4">Model Uncertainty (σ)</th>
                <th className="py-3 px-4">QED</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] bg-black/20 font-mono">
              {rankedUnassayed.map((c: any, index: number) => {
                const isSelected = selectedCandidates.includes(c.id);
                const isTop1 = index === 0;

                return (
                  <tr 
                    key={c.id} 
                    onClick={() => toggleSelect(c.id)}
                    className={`hover:bg-white/[0.03] cursor-pointer transition-colors ${
                      isSelected ? 'bg-white/[0.06]' : ''
                    } ${isTop1 ? 'border-l-2 border-white' : ''}`}
                  >
                    <td className="py-3 px-3 text-center" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => toggleSelect(c.id)}
                        className="text-slate-400 hover:text-white transition-colors cursor-pointer"
                        title={isSelected ? 'Remove from queue' : 'Add to queue'}
                      >
                        {isSelected ? (
                          <CheckSquare className="w-4 h-4 text-white" />
                        ) : (
                          <Square className="w-4 h-4 text-slate-600" />
                        )}
                      </button>
                    </td>
                    <td className="py-3 px-4 font-sans font-medium text-white">
                      <div className="flex items-center gap-2">
                        <span>{c.name}</span>
                        {isTop1 && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-white/[0.08] text-white border border-white/20">
                            ★ Top Recommendation
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-slate-400 font-mono">{c.id}</div>
                    </td>
                    <td className="py-3 px-4 font-semibold text-white text-sm">
                      {c.acquisition_score !== null ? c.acquisition_score : '—'}
                    </td>
                    <td className="py-3 px-4 text-slate-200">
                      {c.predicted_value !== null && c.predicted_value !== undefined ? c.predicted_value : '—'}
                    </td>
                    <td className="py-3 px-4 text-amber-300">
                      ±{c.uncertainty_sigma !== null ? c.uncertainty_sigma : '—'}
                    </td>
                    <td className="py-3 px-4 text-emerald-400 font-semibold">
                      {c.qed}
                    </td>
                    <td className="py-3 px-4 font-sans text-[11px]">
                      <span className="px-2.5 py-0.5 rounded-full bg-black/40 border border-white/[0.06] text-slate-400 font-mono">
                        Unassayed
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-sans">
                      {onSelectCandidate && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectCandidate(c.smiles);
                          }}
                          className="px-3 py-1 bg-white/[0.05] hover:bg-white hover:text-zinc-950 text-slate-300 rounded-full text-xs transition-all cursor-pointer font-medium"
                        >
                          Inspect
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
};
