/**
 * DiscoveryModule — module 04.
 *
 * Four uncertainty-aware decision tools behind one index: the probabilistic
 * Pareto front, the risk slider, the EHVI budget planner, and the generator.
 *
 * Each panel renders its own caveats from the backend payload rather than from
 * hardcoded copy, so when an assumption fails — as the Gaussian shape
 * assumption does on this data — the warning appears automatically.
 */

import React, { useCallback, useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { AnimatePresence, motion } from 'framer-motion';
import { Dice5, Loader2, Sliders, Sparkles, TriangleAlert, Wallet } from 'lucide-react';
import { CountUp, DUR, EASE, Magnetic, RevealText, Rise, Stagger, StaggerItem } from './primitives';
import { useScene } from './SceneCanvas';
import { PLOT_COLORS, PLOT_CONFIG, plotLayout } from './plotTheme';

type Panel = 'probabilistic' | 'risk' | 'planner' | 'generator';

const PANELS: { id: Panel; label: string; icon: React.ElementType; blurb: string }[] = [
  { id: 'probabilistic', label: 'Probabilistic Pareto', icon: Dice5, blurb: 'How likely is each candidate to be a real choice?' },
  { id: 'risk', label: 'Risk slider', icon: Sliders, blurb: 'Trade playing safe against taking a chance.' },
  { id: 'planner', label: 'Budget planner', icon: Wallet, blurb: 'Which experiments buy the most improvement?' },
  { id: 'generator', label: 'Generator', icon: Sparkles, blurb: 'Invent new structures and screen them.' },
];

const Caveat: React.FC<{ children: React.ReactNode; warn?: boolean }> = ({ children, warn }) => (
  <div
    className="mt-5 p-4 rounded-2xl text-[11px] leading-relaxed flex gap-3"
    style={{
      background: warn ? 'var(--pp-caution-wash)' : 'var(--pp-sunken)',
      border: `1px solid ${warn ? 'rgba(180,83,15,0.25)' : 'var(--in-line)'}`,
      color: warn ? 'rgba(240,179,91,0.9)' : 'var(--in-dim)',
    }}
  >
    <TriangleAlert
      className="w-3.5 h-3.5 shrink-0 mt-0.5"
      style={{ color: warn ? 'var(--in-caution)' : 'var(--in-faint)' }}
    />
    <div>{children}</div>
  </div>
);

const Busy: React.FC<{ label: string }> = ({ label }) => (
  <div className="flex flex-col items-center gap-4 py-20">
    <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--in-signal)' }} />
    <span className="in-label">{label}</span>
  </div>
);

export const DiscoveryModule: React.FC = () => {
  const scene = useScene();
  const [panel, setPanel] = useState<Panel>('probabilistic');

  // --- probabilistic Pareto ------------------------------------------------
  const [prob, setProb] = useState<any | null>(null);
  const [probBusy, setProbBusy] = useState(false);
  const [probError, setProbError] = useState<string | null>(null);

  const loadProb = useCallback(async () => {
    setProbBusy(true);
    setProbError(null);
    try {
      const res = await fetch('/api/pareto/probabilistic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_samples: 500 }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Request failed.');
      setProb(json);
    } catch (e) {
      setProbError(e instanceof Error ? e.message : 'Request failed.');
    } finally {
      setProbBusy(false);
    }
  }, []);

  // --- risk slider ---------------------------------------------------------
  const [lambda, setLambda] = useState(0);
  const [risk, setRisk] = useState<any | null>(null);
  const [riskBusy, setRiskBusy] = useState(false);

  const loadRisk = useCallback(async (lam: number) => {
    setRiskBusy(true);
    try {
      const res = await fetch('/api/pareto/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lam }),
      });
      const json = await res.json();
      if (res.ok) setRisk(json);
    } finally {
      setRiskBusy(false);
    }
  }, []);

  useEffect(() => {
    if (panel !== 'risk') return;
    const t = window.setTimeout(() => loadRisk(lambda), 220);
    return () => window.clearTimeout(t);
  }, [panel, lambda, loadRisk]);

  // --- planner -------------------------------------------------------------
  const [budget, setBudget] = useState(8);
  const [assays, setAssays] = useState<any[]>([]);
  const [plan, setPlan] = useState<any | null>(null);
  const [planBusy, setPlanBusy] = useState(false);
  const [ehviCheck, setEhviCheck] = useState<any | null>(null);

  const loadPlan = useCallback(async (b: number) => {
    setPlanBusy(true);
    try {
      const res = await fetch('/api/planner/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ budget: b }),
      });
      const json = await res.json();
      if (res.ok) setPlan(json);
    } finally {
      setPlanBusy(false);
    }
  }, []);

  // --- generator -----------------------------------------------------------
  const [genStatus, setGenStatus] = useState<any | null>(null);
  const [gen, setGen] = useState<any | null>(null);
  const [genBusy, setGenBusy] = useState(false);
  const [temperature, setTemperature] = useState(0.9);

  const runGen = useCallback(async (t: number) => {
    setGenBusy(true);
    try {
      const res = await fetch('/api/generator/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n: 150, temperature: t, top_k: 12 }),
      });
      const json = await res.json();
      if (res.ok) setGen(json);
    } finally {
      setGenBusy(false);
    }
  }, []);

  // Lazy-load each panel's data the first time it is opened.
  useEffect(() => {
    scene.setMode('discovery');
    scene.setUncertainty(0.3);
    scene.setOutOfDomain(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (panel === 'probabilistic' && !prob && !probBusy) loadProb();
    if (panel === 'planner') {
      if (!assays.length) {
        fetch('/api/planner/assays')
          .then((r) => (r.ok ? r.json() : null))
          .then((j) => j && setAssays(j.assays))
          .catch(() => undefined);
      }
      if (!plan && !planBusy) loadPlan(budget);
      if (!ehviCheck) {
        fetch('/api/planner/validate-ehvi')
          .then((r) => (r.ok ? r.json() : null))
          .then(setEhviCheck)
          .catch(() => undefined);
      }
    }
    if (panel === 'generator' && !genStatus) {
      fetch('/api/generator/status')
        .then((r) => (r.ok ? r.json() : null))
        .then(setGenStatus)
        .catch(() => undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel]);

  const active = PANELS.find((p) => p.id === panel)!;

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      <div className="mb-12">
        <div className="in-label mb-4">module 04 — discovery engine</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="Decide under doubt" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p className="mt-5 text-sm max-w-xl leading-relaxed" style={{ color: 'var(--in-dim)' }}>
            Four tools that take uncertainty as input rather than ignoring it. Each states its
            own assumptions, and says so when one of them fails.
          </p>
        </Rise>
      </div>

      {/* Panel index */}
      <Rise delay={0.15}>
        <div className="flex flex-wrap gap-2 mb-2">
          {PANELS.map((p) => {
            const Icon = p.icon;
            const on = p.id === panel;
            return (
              <button
                key={p.id}
                onClick={() => setPanel(p.id)}
                className="in-btn px-4 py-2.5 cursor-pointer flex items-center gap-2"
                style={{
                  background: on ? 'var(--in-signal)' : 'var(--pp-sunken)',
                  color: on ? '#FFFFFF' : 'var(--in-dim)',
                  border: `1px solid ${on ? 'var(--in-signal)' : 'var(--in-line)'}`,
                  fontWeight: on ? 600 : 400,
                }}
              >
                <Icon className="w-3.5 h-3.5" />
                {p.label}
              </button>
            );
          })}
        </div>
        <p className="in-label mb-6">{active.blurb}</p>
      </Rise>

      <AnimatePresence mode="wait">
        <motion.div
          key={panel}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: DUR.fast, ease: EASE }}
        >
          {/* ======================= PROBABILISTIC ======================= */}
          {panel === 'probabilistic' && (
            <div className="in-panel p-7">
              {probBusy && <Busy label="sampling joint outcomes" />}
              {probError && (
                <p className="text-xs" style={{ color: 'var(--in-alarm)' }}>{probError}</p>
              )}

              {prob && !probBusy && (
                <>
                  <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                    {[
                      { l: 'deterministic front', v: prob.deterministic_front_size, d: 0 },
                      { l: 'expected front size', v: prob.expected_front_size, d: 2, accent: true },
                      { l: 'samples', v: prob.n_samples, d: 0 },
                      { l: 'pre-filtered', v: prob.diagnostics.n_prefiltered, d: 0 },
                    ].map((s) => (
                      <StaggerItem key={s.l}>
                        <div className="p-5 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                          <div className="in-label">{s.l}</div>
                          <div
                            className="in-num text-2xl font-light mt-2"
                            style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}
                          >
                            <CountUp value={Number(s.v)} decimals={s.d} />
                          </div>
                        </div>
                      </StaggerItem>
                    ))}
                  </Stagger>

                  {prob.objectives.length >= 2 && (
                    <div className="h-[420px] mb-5">
                      <Plot
                        data={[
                          {
                            x: prob.candidates.map((c: any) => c[prob.objectives[0]]),
                            y: prob.candidates.map((c: any) => c[prob.objectives[1]]),
                            error_x: {
                              type: 'data',
                              array: prob.candidates.map(
                                (c: any) => c[`${prob.objectives[0]}_sigma_tilde`]
                              ),
                              visible: true,
                              color: 'rgba(23,23,26,0.22)',
                              thickness: 1,
                              width: 2,
                            },
                            error_y: {
                              type: 'data',
                              array: prob.candidates.map(
                                (c: any) => c[`${prob.objectives[1]}_sigma_tilde`]
                              ),
                              visible: true,
                              color: 'rgba(23,23,26,0.22)',
                              thickness: 1,
                              width: 2,
                            },
                            text: prob.candidates.map(
                              (c: any) =>
                                `<b>${c.name}</b><br>P(front) = ${c.probability_pareto_optimal}` +
                                `<br>rank ${c.deterministic_pareto_rank} · ${c.ad_status}`
                            ),
                            hoverinfo: 'text',
                            mode: 'markers',
                            type: 'scatter',
                            marker: {
                              size: 13,
                              color: prob.candidates.map((c: any) => c.probability_pareto_optimal),
                              colorscale: [
                                [0, '#D8D7D0'],
                                [0.5, '#8E8EE8'],
                                [1, PLOT_COLORS.signal],
                              ],
                              cmin: 0,
                              cmax: 1,
                              showscale: true,
                              colorbar: {
                                title: { text: 'P(front)', side: 'right', font: { size: 10 } },
                                thickness: 10,
                                tickfont: { size: 9, family: 'JetBrains Mono, monospace' },
                              },
                              line: { color: 'rgba(23,23,26,0.25)', width: 1 },
                            },
                          },
                        ] as any}
                        layout={plotLayout({ x: prob.objectives[0], y: prob.objectives[1] }) as any}
                        config={PLOT_CONFIG as any}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  )}

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                          {['candidate', 'P(on the front)', 'deterministic rank', 'domain'].map((h) => (
                            <th key={h} className="in-label text-left py-3 px-3">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {prob.candidates.slice(0, 10).map((c: any) => (
                          <tr key={c.id} style={{ borderBottom: '1px solid var(--pp-rule-soft)' }}>
                            <td className="py-2.5 px-3">{c.name}</td>
                            <td className="py-2.5 px-3">
                              <div className="flex items-center gap-2.5">
                                <div
                                  className="h-1 rounded-full"
                                  style={{
                                    width: `${c.probability_pareto_optimal * 70}px`,
                                    minWidth: 2,
                                    background: 'var(--in-signal)',
                                  }}
                                />
                                <span className="in-num" style={{ color: 'var(--in-signal)' }}>
                                  {c.probability_pareto_optimal.toFixed(3)}
                                </span>
                              </div>
                            </td>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                              {c.deterministic_pareto_rank}
                              {c.is_deterministic_pareto_optimal ? ' ★' : ''}
                            </td>
                            <td className="py-2.5 px-3 in-label">{c.ad_status}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {prob.residual_normality?.gaussian_approximation_adequate === false && (
                    <Caveat warn>
                      <strong>The bell-curve assumption does not hold on this data.</strong>{' '}
                      Calibration residuals have excess kurtosis{' '}
                      {prob.residual_normality.excess_kurtosis} — heavier tails than the sampler
                      assumes. The calibrated interval <em>width</em> is still valid, but these
                      probabilities are <strong>optimistic</strong> for candidates whose
                      membership depends on a rare large error.
                    </Caveat>
                  )}

                  <Caveat>
                    {prob.interpretation}
                    <ul className="list-disc list-inside mt-2 space-y-0.5">
                      {prob.diagnostics.approximations.map((a: string, i: number) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </Caveat>
                </>
              )}
            </div>
          )}

          {/* ============================ RISK =========================== */}
          {panel === 'risk' && (
            <div className="in-panel p-7">
              <div className="mb-8">
                <div className="flex items-baseline justify-between mb-3">
                  <span className="in-label">
                    risk appetite λ = <span style={{ color: 'var(--in-text)' }}>{lambda.toFixed(2)}</span>
                  </span>
                  {risk && (
                    <span className="in-num text-[11px]" style={{ color: 'var(--in-signal)' }}>
                      κ = {risk.kappa.toFixed(2)} of {risk.kappa_max}
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
                  className="w-full cursor-pointer"
                  style={{ accentColor: 'var(--in-signal)' }}
                />
                <div className="flex justify-between in-label mt-2">
                  <span>0 — play safe, rank by the prediction</span>
                  <span>1 — take a chance, reward uncertainty</span>
                </div>
              </div>

              {!risk && <Busy label="ranking candidates" />}

              {risk && (
                <>
                  <div className="in-label mb-4">
                    risk-adjusted front: {risk.risk_pareto_front_size} candidates
                    {riskBusy && <span className="ml-2" style={{ color: 'var(--in-faint)' }}>updating…</span>}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                          {['#', 'candidate', 'score', 'rank', ...risk.objectives.map((o: string) => o.replace(/_/g, ' '))].map(
                            (h) => (
                              <th key={h} className="in-label text-left py-3 px-3 whitespace-nowrap">{h}</th>
                            )
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        <AnimatePresence initial={false}>
                          {risk.candidates.slice(0, 10).map((c: any, i: number) => (
                            <motion.tr
                              key={c.id}
                              layout
                              transition={{ duration: DUR.base, ease: EASE }}
                              style={{ borderBottom: '1px solid var(--pp-rule-soft)' }}
                            >
                              <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-faint)' }}>
                                {String(i + 1).padStart(2, '0')}
                              </td>
                              <td className="py-2.5 px-3">{c.name}</td>
                              <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-signal)' }}>
                                {c.risk_composite_score}
                              </td>
                              <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                                {c.risk_pareto_rank}
                              </td>
                              {risk.objectives.map((o: string) => (
                                <td key={o} className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                                  {c.risk_adjusted_scores[o]}
                                </td>
                              ))}
                            </motion.tr>
                          ))}
                        </AnimatePresence>
                      </tbody>
                    </table>
                  </div>
                  <Caveat warn>{risk.note}</Caveat>
                </>
              )}
            </div>
          )}

          {/* ========================== PLANNER ========================== */}
          {panel === 'planner' && (
            <div className="in-panel p-7">
              {assays.length > 0 && (
                <div className="mb-6">
                  <div className="in-label mb-3">assay catalogue — all simulated</div>
                  <div className="flex flex-wrap gap-2">
                    {assays.map((a) => (
                      <span
                        key={a.name}
                        title={a.description}
                        className="in-btn px-3 py-1.5"
                        style={{ border: '1px solid var(--in-line)', color: 'var(--in-dim)' }}
                      >
                        {a.name.replace(/_/g, ' ')} · cost {a.cost}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex flex-col sm:flex-row sm:items-end gap-5 mb-7">
                <div className="flex-1">
                  <div className="flex justify-between mb-2">
                    <span className="in-label">total budget</span>
                    <span className="in-num text-[11px]" style={{ color: 'var(--in-signal)' }}>
                      {budget}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={40}
                    step={0.5}
                    value={budget}
                    onChange={(e) => setBudget(parseFloat(e.target.value))}
                    className="w-full cursor-pointer"
                    style={{ accentColor: 'var(--in-signal)' }}
                  />
                </div>
                <Magnetic strength={4}>
                  <button
                    onClick={() => loadPlan(budget)}
                    disabled={planBusy}
                    className="in-btn px-6 py-3 cursor-pointer disabled:opacity-40 whitespace-nowrap"
                    style={{ background: 'var(--in-signal)', color: '#FFFFFF', fontWeight: 600 }}
                  >
                    {planBusy ? 'planning…' : 'build plan'}
                  </button>
                </Magnetic>
              </div>

              {planBusy && !plan && <Busy label="evaluating EHVI over molecule-assay pairs" />}

              {plan && (
                <>
                  <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                    {[
                      { l: 'experiments', v: plan.n_selected, d: 0 },
                      { l: 'cost spent', v: plan.total_cost, d: 1, accent: true },
                      { l: 'budget used', v: plan.budget_utilisation * 100, d: 0, u: '%' },
                      { l: 'expected gain', v: plan.expected_hypervolume_gain, d: 4 },
                    ].map((s) => (
                      <StaggerItem key={s.l}>
                        <div className="p-5 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                          <div className="in-label">{s.l}</div>
                          <div
                            className="in-num text-2xl font-light mt-2"
                            style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}
                          >
                            <CountUp value={Number(s.v)} decimals={s.d} />
                            {s.u && <span className="text-[11px]">{s.u}</span>}
                          </div>
                        </div>
                      </StaggerItem>
                    ))}
                  </Stagger>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                          {['#', 'molecule', 'assay', 'cost', 'EHVI', 'per cost', 'cumulative'].map((h) => (
                            <th key={h} className="in-label text-left py-3 px-3 whitespace-nowrap">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {plan.selected.map((e: any, i: number) => (
                          <tr key={`${e.candidate_id}-${e.assay}`} style={{ borderBottom: '1px solid var(--pp-rule-soft)' }}>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-faint)' }}>
                              {String(i + 1).padStart(2, '0')}
                            </td>
                            <td className="py-2.5 px-3">{e.name || e.candidate_id}</td>
                            <td className="py-2.5 px-3" style={{ color: 'var(--in-dim)' }}>
                              {e.assay.replace(/_/g, ' ')}
                            </td>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>{e.cost}</td>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-signal)' }}>
                              {e.ehvi.toFixed(4)}
                              <span className="ml-1" style={{ color: 'var(--in-faint)', fontSize: 9 }}>
                                ±{e.ehvi_standard_error.toFixed(4)}
                              </span>
                            </td>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                              {e.ehvi_per_unit_cost.toFixed(4)}
                            </td>
                            <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-faint)' }}>
                              {e.cumulative_cost}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {plan.selected.length === 0 && (
                    <p className="text-xs py-8 text-center" style={{ color: 'var(--in-dim)' }}>
                      Nothing selected. {plan.stopped_because}
                    </p>
                  )}

                  {/* The cross-check against BoTorch, shown live */}
                  {ehviCheck?.status === 'ok' && (
                    <div
                      className="mt-5 p-5 rounded-2xl"
                      style={{ background: 'var(--in-signal-dim)', border: '1px solid rgba(76,224,224,0.2)' }}
                    >
                      <div className="in-label" style={{ color: 'var(--in-signal)' }}>
                        verified against BoTorch
                      </div>
                      <p className="text-[11px] mt-2 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                        Our Monte Carlo estimator and BoTorch's own qEHVI, integrating the same
                        posterior, differ by at most{' '}
                        <strong style={{ color: 'var(--in-signal)' }}>
                          {(ehviCheck.max_relative_difference * 100).toFixed(2)}%
                        </strong>
                        . A weighted mu + k·sigma heuristic could not reproduce that.
                      </p>
                    </div>
                  )}

                  <Caveat>
                    <strong>Stopped:</strong> {plan.stopped_because}
                    <div className="mt-1.5">{plan.mode_note}</div>
                    <ul className="list-disc list-inside mt-2 space-y-0.5">
                      {plan.diagnostics.limitations.slice(0, 3).map((l: string, i: number) => (
                        <li key={i}>{l}</li>
                      ))}
                    </ul>
                  </Caveat>
                </>
              )}
            </div>
          )}

          {/* ========================= GENERATOR ========================= */}
          {panel === 'generator' && (
            <div className="in-panel p-7">
              {genStatus && !['prior_trained', 'rl_trained'].includes(genStatus.status) ? (
                <p className="text-xs" style={{ color: 'var(--in-alarm)' }}>
                  Generator unavailable: {genStatus.reason || genStatus.status}
                </p>
              ) : (
                <>
                  {genStatus?.prior?.data_limitation && (
                    <Caveat warn>
                      <strong>MOSES pre-training is pending.</strong>{' '}
                      {genStatus.prior.data_limitation.reason}{' '}
                      {genStatus.prior.data_limitation.expected_effect}
                    </Caveat>
                  )}

                  <div className="flex flex-col sm:flex-row sm:items-end gap-5 my-7">
                    <div className="flex-1">
                      <div className="flex justify-between mb-2">
                        <span className="in-label">sampling temperature</span>
                        <span className="in-num text-[11px]" style={{ color: 'var(--in-signal)' }}>
                          {temperature.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0.5}
                        max={1.5}
                        step={0.05}
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full cursor-pointer"
                        style={{ accentColor: 'var(--in-signal)' }}
                      />
                      <div className="flex justify-between in-label mt-2">
                        <span>0.5 — cautious, more valid</span>
                        <span>1.5 — adventurous, less valid</span>
                      </div>
                    </div>
                    <Magnetic strength={4}>
                      <button
                        onClick={() => runGen(temperature)}
                        disabled={genBusy}
                        className="in-btn px-6 py-3 cursor-pointer disabled:opacity-40 whitespace-nowrap"
                        style={{ background: 'var(--in-signal)', color: '#FFFFFF', fontWeight: 600 }}
                      >
                        {genBusy ? 'generating…' : 'generate 150'}
                      </button>
                    </Magnetic>
                  </div>

                  {genBusy && <Busy label="sampling and screening molecules" />}

                  {gen && !genBusy && (
                    <>
                      <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                        {[
                          { l: 'validity', v: gen.generation_metrics.validity },
                          { l: 'uniqueness', v: gen.generation_metrics.uniqueness },
                          { l: 'novelty', v: gen.generation_metrics.novelty },
                          { l: 'diversity', v: gen.generation_metrics.internal_diversity },
                        ].map((s) => (
                          <StaggerItem key={s.l}>
                            <div className="p-5 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                              <div className="in-label">{s.l}</div>
                              <div className="in-num text-2xl font-light mt-2">
                                {s.v == null ? '—' : <CountUp value={Number(s.v)} decimals={3} />}
                              </div>
                            </div>
                          </StaggerItem>
                        ))}
                      </Stagger>

                      <div className="overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                              {['SMILES', 'logS', '90% interval', 'QED', 'domain', 'novel'].map((h) => (
                                <th key={h} className="in-label text-left py-3 px-3">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {gen.top_candidates.map((c: any) => (
                              <tr key={c.smiles} style={{ borderBottom: '1px solid var(--pp-rule-soft)' }}>
                                <td className="py-2.5 px-3 in-num max-w-[240px] truncate" style={{ color: 'var(--in-text)' }}>
                                  {c.smiles}
                                </td>
                                <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-signal)' }}>
                                  {c.solubility_logs}
                                </td>
                                <td className="py-2.5 px-3 in-num text-[10px]" style={{ color: 'var(--in-faint)' }}>
                                  [{c.solubility_lower_90}, {c.solubility_upper_90}]
                                </td>
                                <td className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>{c.qed}</td>
                                <td className="py-2.5 px-3 in-label">{c.ad_status}</td>
                                <td className="py-2.5 px-3 in-label" style={{ color: c.novel ? 'var(--in-signal)' : 'var(--in-faint)' }}>
                                  {c.novel ? 'yes' : 'no'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <Caveat warn>{gen.ranking_basis}</Caveat>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
};
