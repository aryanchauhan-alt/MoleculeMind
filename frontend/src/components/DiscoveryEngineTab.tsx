import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import {
  Dice5,
  Sliders,
  Wallet,
  Sparkles,
  AlertTriangle,
  Loader2,
  Info,
  TrendingUp,
} from 'lucide-react';

type Panel = 'probabilistic' | 'risk' | 'planner' | 'generator';

const PANELS: { key: Panel; label: string; icon: React.ElementType; blurb: string }[] = [
  {
    key: 'probabilistic',
    label: 'Probabilistic Pareto',
    icon: Dice5,
    blurb: 'Probability each candidate is non-dominated, under its calibrated uncertainty.',
  },
  {
    key: 'risk',
    label: 'Risk Slider',
    icon: Sliders,
    blurb: 'Trade exploitation against exploration with a live λ.',
  },
  {
    key: 'planner',
    label: 'Budget Planner',
    icon: Wallet,
    blurb: 'Cost-aware experiment plan maximising expected hypervolume improvement.',
  },
  {
    key: 'generator',
    label: 'Molecular Generator',
    icon: Sparkles,
    blurb: 'Sample new structures and screen them through the platform models.',
  },
];

const Caveat: React.FC<{ children: React.ReactNode; tone?: 'amber' | 'slate' }> = ({
  children,
  tone = 'slate',
}) => (
  <div
    className={`mt-4 p-4 rounded-2xl border text-[11px] leading-relaxed flex gap-2.5 ${
      tone === 'amber'
        ? 'bg-amber-500/5 border-amber-500/20 text-amber-200/90'
        : 'bg-black/30 border-white/[0.06] text-slate-400'
    }`}
  >
    {tone === 'amber' ? (
      <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
    ) : (
      <Info className="w-4 h-4 shrink-0 text-slate-500 mt-0.5" />
    )}
    <div>{children}</div>
  </div>
);

const Spinner: React.FC<{ label: string }> = ({ label }) => (
  <div className="flex flex-col items-center justify-center h-64 text-slate-400 gap-3">
    <Loader2 className="w-6 h-6 animate-spin" />
    <p className="text-xs font-mono">{label}</p>
  </div>
);

const ErrorBox: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="p-5 rounded-2xl bg-rose-500/5 border border-rose-500/20 text-rose-200 text-xs flex flex-col gap-3">
    <div className="flex gap-2.5">
      <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
      <span className="leading-relaxed">{message}</span>
    </div>
    {onRetry && (
      <button
        onClick={onRetry}
        className="self-start px-4 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white border border-white/20 transition-colors cursor-pointer"
      >
        Retry
      </button>
    )}
  </div>
);

const PLOT_LAYOUT_BASE: any = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(8, 9, 14, 0.8)',
  font: { color: '#e2e8f0', family: 'ui-monospace, monospace', size: 11 },
  margin: { l: 60, r: 20, t: 20, b: 50 },
  showlegend: true,
  legend: { orientation: 'h', y: -0.22 },
};

export const DiscoveryEngineTab: React.FC = () => {
  const [panel, setPanel] = useState<Panel>('probabilistic');

  // --- probabilistic Pareto -------------------------------------------------
  const [probData, setProbData] = useState<any | null>(null);
  const [probLoading, setProbLoading] = useState(false);
  const [probError, setProbError] = useState<string | null>(null);

  const loadProbabilistic = useCallback(async () => {
    setProbLoading(true);
    setProbError(null);
    try {
      const res = await fetch('/api/pareto/probabilistic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 500 }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Request failed.');
      setProbData(json);
    } catch (e) {
      setProbError(e instanceof Error ? e.message : 'Request failed.');
    } finally {
      setProbLoading(false);
    }
  }, []);

  // --- risk slider ----------------------------------------------------------
  const [lambda, setLambda] = useState(0);
  const [riskData, setRiskData] = useState<any | null>(null);
  const [riskLoading, setRiskLoading] = useState(false);
  const [riskError, setRiskError] = useState<string | null>(null);

  const loadRisk = useCallback(async (lam: number) => {
    setRiskLoading(true);
    setRiskError(null);
    try {
      const res = await fetch('/api/pareto/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lam }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Request failed.');
      setRiskData(json);
    } catch (e) {
      setRiskError(e instanceof Error ? e.message : 'Request failed.');
    } finally {
      setRiskLoading(false);
    }
  }, []);

  // Debounced so dragging the slider does not flood the backend.
  useEffect(() => {
    if (panel !== 'risk') return;
    const timer = window.setTimeout(() => loadRisk(lambda), 250);
    return () => window.clearTimeout(timer);
  }, [panel, lambda, loadRisk]);

  // --- planner --------------------------------------------------------------
  const [budget, setBudget] = useState(8);
  const [assays, setAssays] = useState<any[]>([]);
  const [plan, setPlan] = useState<any | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const loadPlan = useCallback(async (b: number) => {
    setPlanLoading(true);
    setPlanError(null);
    try {
      const res = await fetch('/api/planner/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ budget: b }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Request failed.');
      setPlan(json);
    } catch (e) {
      setPlanError(e instanceof Error ? e.message : 'Request failed.');
    } finally {
      setPlanLoading(false);
    }
  }, []);

  // --- generator ------------------------------------------------------------
  const [genStatus, setGenStatus] = useState<any | null>(null);
  const [genResult, setGenResult] = useState<any | null>(null);
  const [genLoading, setGenLoading] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(1.0);

  const runGeneration = useCallback(async (temp: number) => {
    setGenLoading(true);
    setGenError(null);
    try {
      const res = await fetch('/api/generator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n: 150, temperature: temp, top_k: 12 }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Request failed.');
      setGenResult(json);
    } catch (e) {
      setGenError(e instanceof Error ? e.message : 'Request failed.');
    } finally {
      setGenLoading(false);
    }
  }, []);

  useEffect(() => {
    if (panel === 'probabilistic' && !probData && !probLoading) loadProbabilistic();
    if (panel === 'planner') {
      if (!assays.length) {
        fetch('/api/planner/assays')
          .then((r) => (r.ok ? r.json() : null))
          .then((j) => j && setAssays(j.assays))
          .catch(() => undefined);
      }
      if (!plan && !planLoading) loadPlan(budget);
    }
    if (panel === 'generator' && !genStatus) {
      fetch('/api/generator/status')
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => j && setGenStatus(j))
        .catch(() => undefined);
    }
  }, [panel]); // eslint-disable-line react-hooks/exhaustive-deps

  const active = PANELS.find((p) => p.key === panel)!;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Panel selector */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-5 sm:p-6 backdrop-blur-2xl">
        <div className="flex items-center gap-2.5 mb-1">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-medium text-white">Discovery Engine</h2>
        </div>
        <p className="text-xs text-slate-400 mb-5">
          Uncertainty-aware decision tools. Every number here derives from a calibrated
          predictive distribution, and each panel states its own assumptions.
        </p>

        <div className="flex flex-wrap gap-2">
          {PANELS.map((p) => {
            const Icon = p.icon;
            const isActive = p.key === panel;
            return (
              <button
                key={p.key}
                onClick={() => setPanel(p.key)}
                className={`px-4 py-2 rounded-full text-xs font-medium transition-all cursor-pointer flex items-center gap-2 border ${
                  isActive
                    ? 'bg-white text-zinc-950 border-white font-semibold'
                    : 'bg-white/[0.03] text-slate-300 border-white/[0.08] hover:bg-white/[0.08] hover:text-white'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {p.label}
              </button>
            );
          })}
        </div>
        <p className="text-[11px] text-slate-500 mt-3 font-mono">{active.blurb}</p>
      </div>

      {/* ============================ Probabilistic Pareto ==================== */}
      {panel === 'probabilistic' && (
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 backdrop-blur-2xl">
          {probLoading && <Spinner label="Sampling joint objective vectors..." />}
          {probError && <ErrorBox message={probError} onRetry={loadProbabilistic} />}

          {probData && !probLoading && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 text-xs">
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Deterministic front</div>
                  <div className="text-2xl font-light font-mono text-white mt-1">
                    {probData.deterministic_front_size}
                  </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Expected front size</div>
                  <div className="text-2xl font-light font-mono text-cyan-300 mt-1">
                    {probData.expected_front_size}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">sum of probabilities</div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">MC samples</div>
                  <div className="text-2xl font-light font-mono text-white mt-1">
                    {probData.n_samples}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    max SE {probData.diagnostics.max_monte_carlo_standard_error}
                  </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Pre-filtered</div>
                  <div className="text-2xl font-light font-mono text-slate-300 mt-1">
                    {probData.diagnostics.n_prefiltered}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    k = {probData.diagnostics.prefilter_k}
                  </div>
                </div>
              </div>

              {/* Probability-coloured scatter over the first two objectives */}
              {probData.objectives.length >= 2 && (
                <div className="h-[420px] mb-4">
                  <Plot
                    data={[
                      {
                        x: probData.candidates.map((c: any) => c[probData.objectives[0]]),
                        y: probData.candidates.map((c: any) => c[probData.objectives[1]]),
                        error_x: {
                          type: 'data',
                          array: probData.candidates.map(
                            (c: any) => c[`${probData.objectives[0]}_sigma_tilde`]
                          ),
                          visible: true,
                          color: 'rgba(148,163,184,0.35)',
                          thickness: 1,
                          width: 2,
                        },
                        error_y: {
                          type: 'data',
                          array: probData.candidates.map(
                            (c: any) => c[`${probData.objectives[1]}_sigma_tilde`]
                          ),
                          visible: true,
                          color: 'rgba(148,163,184,0.35)',
                          thickness: 1,
                          width: 2,
                        },
                        text: probData.candidates.map(
                          (c: any) =>
                            `<b>${c.name}</b><br>P(Pareto) = ${c.probability_pareto_optimal}` +
                            `<br>deterministic rank ${c.deterministic_pareto_rank}` +
                            `<br>AD: ${c.ad_status}`
                        ),
                        hoverinfo: 'text',
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Candidates',
                        marker: {
                          size: 13,
                          color: probData.candidates.map(
                            (c: any) => c.probability_pareto_optimal
                          ),
                          colorscale: 'Viridis',
                          cmin: 0,
                          cmax: 1,
                          showscale: true,
                          colorbar: {
                            title: { text: 'P(Pareto)', side: 'right' },
                            thickness: 12,
                          },
                          line: { color: 'rgba(255,255,255,0.35)', width: 1 },
                        },
                      } as any,
                    ]}
                    layout={{
                      ...PLOT_LAYOUT_BASE,
                      showlegend: false,
                      xaxis: {
                        title: probData.objectives[0],
                        gridcolor: 'rgba(255,255,255,0.06)',
                      },
                      yaxis: {
                        title: probData.objectives[1],
                        gridcolor: 'rgba(255,255,255,0.06)',
                      },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-slate-400 font-mono uppercase tracking-wider text-[10px] border-b border-white/[0.06]">
                    <tr>
                      <th className="py-2.5 px-3 text-left">Candidate</th>
                      <th className="py-2.5 px-3 text-right">P(Pareto optimal)</th>
                      <th className="py-2.5 px-3 text-right">Deterministic rank</th>
                      <th className="py-2.5 px-3 text-left">AD</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {probData.candidates.slice(0, 12).map((c: any) => (
                      <tr key={c.id}>
                        <td className="py-2.5 px-3 text-white">{c.name}</td>
                        <td className="py-2.5 px-3 text-right font-mono">
                          <span
                            className={
                              c.probability_pareto_optimal > 0.5
                                ? 'text-cyan-300'
                                : 'text-slate-400'
                            }
                          >
                            {c.probability_pareto_optimal.toFixed(3)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                          {c.deterministic_pareto_rank}
                          {c.is_deterministic_pareto_optimal ? ' ★' : ''}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">{c.ad_status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {probData.residual_normality &&
                probData.residual_normality.gaussian_approximation_adequate === false && (
                  <Caveat tone="amber">
                    <strong>The Gaussian shape assumption does not hold on this data.</strong>{' '}
                    Calibration residuals have excess kurtosis{' '}
                    {probData.residual_normality.excess_kurtosis} (0 for a Gaussian), so they are
                    heavier-tailed than the sampler assumes. The calibrated interval{' '}
                    <em>width</em> remains valid, but these probabilities are{' '}
                    <strong>optimistic</strong> for candidates whose front membership depends on a
                    tail event.
                  </Caveat>
                )}

              <Caveat>
                {probData.interpretation}
                <div className="mt-2 pt-2 border-t border-white/[0.06]">
                  <span className="text-slate-500">Approximations in force:</span>
                  <ul className="list-disc list-inside mt-1 space-y-0.5">
                    {probData.diagnostics.approximations.map((a: string, i: number) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                </div>
              </Caveat>
            </>
          )}
        </div>
      )}

      {/* ================================ Risk slider ========================= */}
      {panel === 'risk' && (
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 backdrop-blur-2xl">
          <div className="mb-6">
            <div className="flex items-baseline justify-between mb-2">
              <label className="text-xs font-mono text-slate-400">
                Risk appetite λ = <span className="text-white">{lambda.toFixed(2)}</span>
              </label>
              {riskData && (
                <span className="text-[11px] font-mono text-cyan-300">
                  κ = {riskData.kappa.toFixed(2)} (κ<sub>max</sub> = {riskData.kappa_max})
                </span>
              )}
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={lambda}
              onChange={(e) => setLambda(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
              <span>0 — exploit (rank by the mean alone)</span>
              <span>1 — explore (uncertainty at full weight)</span>
            </div>
          </div>

          {riskError && <ErrorBox message={riskError} onRetry={() => loadRisk(lambda)} />}
          {riskLoading && !riskData && <Spinner label="Re-ranking candidates..." />}

          {riskData && (
            <>
              <div className="text-[11px] text-slate-400 font-mono mb-3">
                Risk-adjusted Pareto front: {riskData.risk_pareto_front_size} candidates
                {riskLoading && <span className="ml-2 text-slate-500">updating…</span>}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-slate-400 font-mono uppercase tracking-wider text-[10px] border-b border-white/[0.06]">
                    <tr>
                      <th className="py-2.5 px-3 text-left">#</th>
                      <th className="py-2.5 px-3 text-left">Candidate</th>
                      <th className="py-2.5 px-3 text-right">Composite</th>
                      <th className="py-2.5 px-3 text-right">Rank</th>
                      {riskData.objectives.map((o: string) => (
                        <th key={o} className="py-2.5 px-3 text-right">
                          {o.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {riskData.candidates.slice(0, 12).map((c: any, i: number) => (
                      <tr key={c.id}>
                        <td className="py-2.5 px-3 text-slate-500 font-mono">{i + 1}</td>
                        <td className="py-2.5 px-3 text-white">{c.name}</td>
                        <td className="py-2.5 px-3 text-right font-mono text-cyan-300">
                          {c.risk_composite_score}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                          {c.risk_pareto_rank}
                        </td>
                        {riskData.objectives.map((o: string) => (
                          <td key={o} className="py-2.5 px-3 text-right font-mono text-slate-300">
                            {c.risk_adjusted_scores[o]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Caveat tone="amber">{riskData.note}</Caveat>
            </>
          )}
        </div>
      )}

      {/* ================================= Planner ============================ */}
      {panel === 'planner' && (
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 backdrop-blur-2xl">
          {assays.length > 0 && (
            <div className="mb-5">
              <div className="text-[11px] text-slate-400 font-mono mb-2">
                Assay catalogue (costs in budget units — all assays are simulated)
              </div>
              <div className="flex flex-wrap gap-2">
                {assays.map((a) => (
                  <span
                    key={a.name}
                    title={a.description}
                    className="px-3 py-1.5 rounded-full bg-black/40 border border-white/[0.08] text-[11px] font-mono text-slate-300"
                  >
                    {a.name.replace(/_/g, ' ')} · {a.cost}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-col sm:flex-row sm:items-end gap-4 mb-6">
            <div className="flex-1">
              <label className="text-xs font-mono text-slate-400">
                Total budget B = <span className="text-white">{budget}</span>
              </label>
              <input
                type="range"
                min={1}
                max={40}
                step={0.5}
                value={budget}
                onChange={(e) => setBudget(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer mt-2"
              />
            </div>
            <button
              onClick={() => loadPlan(budget)}
              disabled={planLoading}
              className="px-5 py-2.5 rounded-full bg-white hover:bg-slate-200 disabled:opacity-50 text-zinc-950 font-semibold text-xs transition-all cursor-pointer whitespace-nowrap"
            >
              {planLoading ? 'Planning…' : 'Build plan'}
            </button>
          </div>

          {planError && <ErrorBox message={planError} onRetry={() => loadPlan(budget)} />}
          {planLoading && !plan && <Spinner label="Evaluating EHVI over molecule-assay pairs..." />}

          {plan && (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5 text-xs">
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Experiments</div>
                  <div className="text-2xl font-light font-mono text-white mt-1">
                    {plan.n_selected}
                  </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Cost / budget</div>
                  <div className="text-2xl font-light font-mono text-cyan-300 mt-1">
                    {plan.total_cost}
                    <span className="text-sm text-slate-500"> / {plan.budget}</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    {Math.round(plan.budget_utilisation * 100)}% used
                  </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Expected HV gain</div>
                  <div className="text-2xl font-light font-mono text-emerald-300 mt-1">
                    {plan.expected_hypervolume_gain}
                  </div>
                </div>
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono text-[11px]">Reference point</div>
                  <div className="text-xs font-mono text-slate-300 mt-2">
                    [{plan.reference_point.join(', ')}]
                  </div>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-slate-400 font-mono uppercase tracking-wider text-[10px] border-b border-white/[0.06]">
                    <tr>
                      <th className="py-2.5 px-3 text-left">#</th>
                      <th className="py-2.5 px-3 text-left">Molecule</th>
                      <th className="py-2.5 px-3 text-left">Assay</th>
                      <th className="py-2.5 px-3 text-right">Cost</th>
                      <th className="py-2.5 px-3 text-right">EHVI</th>
                      <th className="py-2.5 px-3 text-right">EHVI / cost</th>
                      <th className="py-2.5 px-3 text-right">Cumulative</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {plan.selected.map((e: any, i: number) => (
                      <tr key={`${e.candidate_id}-${e.assay}`}>
                        <td className="py-2.5 px-3 text-slate-500 font-mono">{i + 1}</td>
                        <td className="py-2.5 px-3 text-white">{e.name || e.candidate_id}</td>
                        <td className="py-2.5 px-3 text-slate-400">{e.assay.replace(/_/g, ' ')}</td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-300">{e.cost}</td>
                        <td className="py-2.5 px-3 text-right font-mono text-cyan-300">
                          {e.ehvi.toFixed(4)}
                          <span className="text-slate-600 text-[10px]">
                            {' '}
                            ±{e.ehvi_standard_error.toFixed(4)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                          {e.ehvi_per_unit_cost.toFixed(4)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                          {e.cumulative_cost}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {plan.selected.length === 0 && (
                <p className="text-xs text-slate-400 py-6 text-center">
                  No experiment was selected. {plan.stopped_because}
                </p>
              )}

              <Caveat>
                <strong>Stopped because:</strong> {plan.stopped_because}
                <div className="mt-2">{plan.mode_note}</div>
                <div className="mt-2 pt-2 border-t border-white/[0.06]">
                  <span className="text-slate-500">Method:</span>{' '}
                  {plan.diagnostics.method}
                  <ul className="list-disc list-inside mt-1.5 space-y-0.5">
                    {plan.diagnostics.limitations.map((l: string, i: number) => (
                      <li key={i}>{l}</li>
                    ))}
                  </ul>
                </div>
              </Caveat>
            </>
          )}
        </div>
      )}

      {/* ================================ Generator =========================== */}
      {panel === 'generator' && (
        <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 backdrop-blur-2xl">
          {genStatus && genStatus.status !== 'prior_trained' && genStatus.status !== 'rl_trained' ? (
            <ErrorBox
              message={`Generator unavailable: ${genStatus.reason || genStatus.status}`}
            />
          ) : (
            <>
              {genStatus?.prior?.data_limitation && (
                <Caveat tone="amber">
                  <strong>MOSES pre-training is pending.</strong>{' '}
                  {genStatus.prior.data_limitation.reason}{' '}
                  {genStatus.prior.data_limitation.expected_effect}
                </Caveat>
              )}

              {genStatus?.architecture && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-5 text-xs">
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono text-[11px]">Architecture</div>
                    <div className="text-slate-200 mt-1 text-[11px]">GRU LM</div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      {genStatus.architecture.n_layers} × {genStatus.architecture.hidden_size}
                    </div>
                  </div>
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono text-[11px]">Vocabulary</div>
                    <div className="text-2xl font-light font-mono text-white mt-1">
                      {genStatus.architecture.vocabulary_size}
                    </div>
                  </div>
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono text-[11px]">Prior corpus</div>
                    <div className="text-2xl font-light font-mono text-white mt-1">
                      {genStatus.prior?.n_training_smiles ?? '—'}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">SMILES</div>
                  </div>
                  <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                    <div className="text-slate-400 font-mono text-[11px]">RL campaign</div>
                    <div className="text-slate-200 mt-1 text-[11px]">
                      {genStatus.reinforcement_learning?.status === 'trained' ? 'run' : 'not run'}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex flex-col sm:flex-row sm:items-end gap-4 mb-6">
                <div className="flex-1">
                  <label className="text-xs font-mono text-slate-400">
                    Sampling temperature = <span className="text-white">{temperature.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min={0.5}
                    max={1.5}
                    step={0.05}
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-cyan-400 cursor-pointer mt-2"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                    <span>0.5 — conservative, higher validity</span>
                    <span>1.5 — adventurous, lower validity</span>
                  </div>
                </div>
                <button
                  onClick={() => runGeneration(temperature)}
                  disabled={genLoading}
                  className="px-5 py-2.5 rounded-full bg-white hover:bg-slate-200 disabled:opacity-50 text-zinc-950 font-semibold text-xs transition-all cursor-pointer whitespace-nowrap"
                >
                  {genLoading ? 'Generating…' : 'Generate 150'}
                </button>
              </div>

              {genError && <ErrorBox message={genError} onRetry={() => runGeneration(temperature)} />}
              {genLoading && <Spinner label="Sampling and screening molecules..." />}

              {genResult && !genLoading && (
                <>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5 text-xs">
                    {[
                      ['Validity', genResult.generation_metrics.validity],
                      ['Uniqueness', genResult.generation_metrics.uniqueness],
                      ['Novelty', genResult.generation_metrics.novelty],
                      ['Internal diversity', genResult.generation_metrics.internal_diversity],
                    ].map(([label, value]) => (
                      <div
                        key={label as string}
                        className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]"
                      >
                        <div className="text-slate-400 font-mono text-[11px]">{label}</div>
                        <div className="text-2xl font-light font-mono text-white mt-1">
                          {value === undefined || value === null ? '—' : value}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead className="text-slate-400 font-mono uppercase tracking-wider text-[10px] border-b border-white/[0.06]">
                        <tr>
                          <th className="py-2.5 px-3 text-left">SMILES</th>
                          <th className="py-2.5 px-3 text-right">logS</th>
                          <th className="py-2.5 px-3 text-left">90% interval</th>
                          <th className="py-2.5 px-3 text-right">QED</th>
                          <th className="py-2.5 px-3 text-left">AD</th>
                          <th className="py-2.5 px-3 text-center">Novel</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/[0.04]">
                        {genResult.top_candidates.map((c: any) => (
                          <tr key={c.smiles}>
                            <td className="py-2.5 px-3 font-mono text-slate-200 max-w-[280px] truncate">
                              {c.smiles}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-cyan-300">
                              {c.solubility_logs}
                            </td>
                            <td className="py-2.5 px-3 font-mono text-slate-500 text-[10px]">
                              [{c.solubility_lower_90}, {c.solubility_upper_90}]
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono text-emerald-300">
                              {c.qed}
                            </td>
                            <td className="py-2.5 px-3 text-slate-400">{c.ad_status}</td>
                            <td className="py-2.5 px-3 text-center">
                              {c.novel ? (
                                <span className="text-cyan-300">yes</span>
                              ) : (
                                <span className="text-slate-600">no</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <Caveat tone="amber">{genResult.ranking_basis}</Caveat>
                </>
              )}
            </>
          )}
        </div>
      )}
    </motion.div>
  );
};
