/**
 * ParetoModule — module 03.
 *
 * Shows the trade-off front over the objectives that actually have a validated
 * model behind them. Objectives without one are listed as excluded rather than
 * quietly dropped, and candidates missing a value are reported rather than
 * imputed — the backend raises on missing values and this surfaces that.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import { Award, Loader2 } from 'lucide-react';
import { CountUp, RevealText, Rise, Stagger, StaggerItem } from './primitives';
import { parseMolBlock, useScene } from './SceneCanvas';
import { PLOT_COLORS, PLOT_CONFIG, plotLayout } from './plotTheme';

const AXES = [
  { key: 'solubility_logs', label: 'Solubility (logS)' },
  { key: 'qed_drug_likeness', label: 'QED drug-likeness' },
  { key: 'toxicity_risk_probability', label: 'Toxicity risk' },
  { key: 'molecular_weight', label: 'Molecular weight' },
  { key: 'ad_score', label: 'Domain index' },
];

export const ParetoModule: React.FC<{ onSelect?: (smiles: string) => void }> = ({ onSelect }) => {
  const scene = useScene();
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [xKey, setXKey] = useState('qed_drug_likeness');
  const [yKey, setYKey] = useState('solubility_logs');
  const [selected, setSelected] = useState<any | null>(null);
  const [frontOnly, setFrontOnly] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/pareto');
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Could not load the candidate pool.');
      setData(json);
      if (json.candidates?.length) setSelected(json.candidates[0]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load the candidate pool.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    scene.setMode('pareto');
    scene.setUncertainty(0.2);
    scene.setOutOfDomain(false);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Show the selected candidate on the shared canvas.
  useEffect(() => {
    if (!selected?.smiles) return;
    let cancelled = false;
    fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles: selected.smiles }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (cancelled || !j?.sdf_3d) return;
        const { atoms, bonds } = parseMolBlock(j.sdf_3d);
        scene.setMolecule(atoms, bonds);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.smiles]);

  const candidates = data?.candidates ?? [];
  const shown = frontOnly ? candidates.filter((c: any) => c.is_pareto_optimal) : candidates;
  const xLabel = AXES.find((a) => a.key === xKey)?.label ?? xKey;
  const yLabel = AXES.find((a) => a.key === yKey)?.label ?? yKey;

  const traces = useMemo(() => {
    const front = shown.filter((c: any) => c.is_pareto_optimal);
    const rest = shown.filter((c: any) => !c.is_pareto_optimal);
    const mk = (pts: any[], name: string, optimal: boolean) => ({
      x: pts.map((c) => c[xKey]),
      y: pts.map((c) => c[yKey]),
      customdata: pts.map((c) => c.id),
      text: pts.map(
        (c) =>
          `<b>${c.name}</b><br>${xLabel}: ${c[xKey]}<br>${yLabel}: ${c[yKey]}` +
          `<br>rank ${c.pareto_rank} · score ${c.composite_score}<br>AD: ${c.ad_status}`
      ),
      hoverinfo: 'text' as const,
      mode: 'markers' as const,
      type: 'scatter' as const,
      name,
      marker: optimal
        ? {
            color: PLOT_COLORS.signal,
            size: 13,
            symbol: 'diamond' as const,
            line: { color: 'rgba(23,23,26,0.35)', width: 1 },
          }
        : { color: PLOT_COLORS.faint, size: 7, opacity: 0.55 },
    });
    return [mk(rest, 'Dominated', false), mk(front, 'Pareto front', true)];
  }, [shown, xKey, yKey, xLabel, yLabel]);

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      <div className="mb-12">
        <div className="in-label mb-4">module 03 — pareto laboratory</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="No single best" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p className="mt-5 text-sm max-w-xl leading-relaxed" style={{ color: 'var(--in-dim)' }}>
            A candidate is dominated when another is at least as good on every objective and
            better on one. What survives is the set of real trade-offs — chosen without
            inventing any weights.
          </p>
        </Rise>
      </div>

      {loading && (
        <div className="in-panel p-16 flex flex-col items-center gap-4">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--in-signal)' }} />
          <span className="in-label">predicting the candidate pool</span>
        </div>
      )}

      {error && (
        <div className="in-panel p-6" style={{ borderColor: 'rgba(179,38,30,0.3)' }}>
          <p className="text-xs" style={{ color: 'var(--in-dim)' }}>{error}</p>
        </div>
      )}

      {data && !loading && (
        <>
          <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
            {[
              { l: 'candidates', v: data.total_candidates, d: 0 },
              { l: 'on the front', v: data.pareto_optimal_count, d: 0, accent: true },
              { l: 'objectives', v: data.objectives.length, d: 0 },
              { l: 'excluded', v: data.excluded_candidates.length, d: 0 },
            ].map((s) => (
              <StaggerItem key={s.l}>
                <div className="in-panel in-panel-hover p-5">
                  <div className="in-label">{s.l}</div>
                  <div
                    className="in-num text-3xl font-light mt-2"
                    style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}
                  >
                    <CountUp value={s.v} decimals={0} />
                  </div>
                </div>
              </StaggerItem>
            ))}
          </Stagger>

          {/* Active objectives, stated explicitly */}
          <Rise delay={0.15}>
            <div className="in-panel p-5 mb-3">
              <div className="in-label mb-3">objectives being optimised</div>
              <div className="flex flex-wrap gap-2">
                {data.objective_specs.map((o: any) => (
                  <span
                    key={o.key}
                    className="in-btn px-3 py-1.5"
                    style={{ border: '1px solid var(--in-line)', color: 'var(--in-text)' }}
                  >
                    {o.label} · {o.maximize ? 'max' : 'min'} · {o.provenance}
                  </span>
                ))}
              </div>
              <p className="text-[10px] mt-3 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
                {data.note}
              </p>
            </div>
          </Rise>

          {/* Plot */}
          <Rise delay={0.2}>
            <div className="in-panel p-6 mb-3">
              <div className="flex flex-wrap items-center gap-3 mb-5">
                {[
                  { label: 'X', value: xKey, set: setXKey },
                  { label: 'Y', value: yKey, set: setYKey },
                ].map((sel) => (
                  <div
                    key={sel.label}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-full"
                    style={{ background: 'var(--pp-sunken)', border: '1px solid var(--in-line)' }}
                  >
                    <span className="in-label">{sel.label}</span>
                    <select
                      value={sel.value}
                      onChange={(e) => sel.set(e.target.value)}
                      className="bg-transparent text-[11px] focus:outline-none cursor-pointer"
                      style={{ color: 'var(--in-text)', fontFamily: 'var(--in-mono)' }}
                    >
                      {AXES.map((a) => (
                        <option key={a.key} value={a.key} style={{ background: '#0C0D14' }}>
                          {a.label}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
                <button
                  onClick={() => setFrontOnly((v) => !v)}
                  className="in-btn px-3.5 py-1.5 cursor-pointer"
                  style={{
                    background: frontOnly ? 'var(--in-signal)' : 'var(--pp-sunken)',
                    color: frontOnly ? '#FFFFFF' : 'var(--in-dim)',
                    border: `1px solid ${frontOnly ? 'var(--in-signal)' : 'var(--in-line)'}`,
                  }}
                >
                  front only
                </button>
              </div>

              <div className="h-[440px]">
                <Plot
                  data={traces as any}
                  layout={{ ...plotLayout({ x: xLabel, y: yLabel, legend: true }) } as any}
                  config={PLOT_CONFIG as any}
                  style={{ width: '100%', height: '100%' }}
                  onClick={(e: any) => {
                    const id = e?.points?.[0]?.customdata;
                    const hit = candidates.find((c: any) => c.id === id);
                    if (hit) setSelected(hit);
                  }}
                />
              </div>
            </div>
          </Rise>

          {/* Selected candidate */}
          {selected && (
            <Rise delay={0.25}>
              <div className="in-panel p-7 mb-3">
                <div className="flex flex-wrap items-start justify-between gap-4 mb-5">
                  <div>
                    <div className="in-label">selected candidate</div>
                    <div className="in-display text-2xl mt-2">{selected.name}</div>
                    <div className="in-num text-[10px] mt-1.5 break-all" style={{ color: 'var(--in-faint)' }}>
                      {selected.smiles}
                    </div>
                  </div>
                  {selected.is_pareto_optimal ? (
                    <span
                      className="in-btn px-3 py-1.5 flex items-center gap-1.5"
                      style={{
                        background: 'var(--in-signal-dim)',
                        color: 'var(--in-signal)',
                        border: '1px solid rgba(76,224,224,0.25)',
                      }}
                    >
                      <Award className="w-3 h-3" /> pareto front
                    </span>
                  ) : (
                    <span className="in-btn px-3 py-1.5" style={{ border: '1px solid var(--in-line)', color: 'var(--in-dim)' }}>
                      rank {selected.pareto_rank}
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {data.objectives.map((k: string) => (
                    <div key={k} className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                      <div className="in-label">{k.replace(/_/g, ' ')}</div>
                      <div className="in-num text-lg mt-1.5" style={{ color: 'var(--in-signal)' }}>
                        {selected[k]}
                      </div>
                    </div>
                  ))}
                  <div className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label">90% interval, logS</div>
                    <div className="in-num text-sm mt-1.5">
                      [{selected.solubility_lower_90}, {selected.solubility_upper_90}]
                    </div>
                  </div>
                </div>

                <p className="text-[11px] mt-5 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                  {selected.is_pareto_optimal
                    ? 'Non-dominated: no other candidate matches it on every active objective and beats it on one. Improving any objective from here means giving ground on another.'
                    : `Dominated: at least one rank-1 candidate matches or beats this on every active objective and beats it on at least one.`}
                </p>

                {onSelect && (
                  <button
                    onClick={() => onSelect(selected.smiles)}
                    className="in-btn px-5 py-2.5 mt-5 cursor-pointer"
                    style={{ border: '1px solid var(--in-line-strong)', color: 'var(--in-text)' }}
                  >
                    inspect in analyzer
                  </button>
                )}
              </div>
            </Rise>
          )}

          {/* Candidate table */}
          <Rise delay={0.3}>
            <div className="in-panel p-6 overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                    {['compound', 'rank', 'score', ...data.objectives.map((o: string) => o.replace(/_/g, ' ')), 'domain'].map(
                      (h) => (
                        <th key={h} className="in-label text-left py-3 px-3 whitespace-nowrap">
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {candidates.slice(0, 14).map((c: any) => (
                    <tr
                      key={c.id}
                      onClick={() => setSelected(c)}
                      className="cursor-pointer transition-colors"
                      style={{
                        borderBottom: '1px solid var(--pp-rule-soft)',
                        background: selected?.id === c.id ? 'var(--pp-accent-wash)' : 'transparent',
                      }}
                    >
                      <td className="py-2.5 px-3">{c.name}</td>
                      <td className="py-2.5 px-3 in-num" style={{ color: c.is_pareto_optimal ? 'var(--in-signal)' : 'var(--in-faint)' }}>
                        {c.is_pareto_optimal ? '★ 1' : c.pareto_rank}
                      </td>
                      <td className="py-2.5 px-3 in-num">{c.composite_score}</td>
                      {data.objectives.map((o: string) => (
                        <td key={o} className="py-2.5 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                          {c[o]}
                        </td>
                      ))}
                      <td className="py-2.5 px-3 in-label">{c.ad_status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Rise>
        </>
      )}
    </div>
  );
};
