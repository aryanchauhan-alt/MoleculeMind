/**
 * BenchmarkModule — module 06.
 *
 * Everything here is computed at startup from a held-out split and fetched, not
 * written into the page. That is a deliberate constraint: an earlier version
 * had R² typed in as a literal, which went stale the moment the model changed
 * and would have contradicted the live app during a demo.
 */

import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import { Download, Loader2 } from 'lucide-react';
import { CountUp, RevealText, Rise, Stagger, StaggerItem } from './primitives';
import { useScene } from './SceneCanvas';
import { PLOT_COLORS, PLOT_CONFIG, plotLayout } from './plotTheme';

export const BenchmarkModule: React.FC = () => {
  const scene = useScene();
  const [data, setData] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    scene.setMode('benchmark');
    scene.setUncertainty(0.15);
    scene.setOutOfDomain(false);
    fetch('/api/benchmark')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('Could not load benchmarks.'))))
      .then(setData)
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
        <div className="in-panel p-6" style={{ borderColor: 'rgba(179,38,30,0.3)' }}>
          <p className="text-xs" style={{ color: 'var(--in-dim)' }}>{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
        <div className="in-panel p-16 flex flex-col items-center gap-4">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--in-signal)' }} />
          <span className="in-label">loading measured performance</span>
        </div>
      </div>
    );
  }

  const m = data.metrics;
  const scaffold = data.scaffold_split_metrics;
  const clintox = data.clintox_metrics;
  const dmpnn = data.dmpnn;
  const parity = data.parity_plot_data ?? [];

  const actual = parity.map((p: any) => p.actual);
  const predicted = parity.map((p: any) => p.predicted);
  const lo = Math.min(...actual, ...predicted) - 0.5;
  const hi = Math.max(...actual, ...predicted) + 0.5;

  const rows = [
    {
      model: 'Extra Trees (baseline)',
      rep: 'ECFP4 + descriptors',
      split: 'random',
      r2: m.r2_score,
      rmse: m.rmse,
      n: m.test_samples,
    },
    scaffold?.r2_score !== undefined && {
      model: 'Extra Trees',
      rep: 'ECFP4 + descriptors',
      split: 'Bemis–Murcko scaffold',
      r2: scaffold.r2_score,
      rmse: scaffold.rmse,
      n: scaffold.test_samples,
      caution: true,
    },
    dmpnn?.status === 'trained' &&
      dmpnn.metrics?.solubility && {
        model: 'D-MPNN ensemble',
        rep: 'molecular graph, learned',
        split: 'random, pinned',
        r2: dmpnn.metrics.solubility.r2_score,
        rmse: dmpnn.metrics.solubility.rmse,
        n: dmpnn.metrics.solubility.n_test,
        accent: true,
      },
    dmpnn?.independent_validation?.status === 'ok' && {
      model: 'Random Forest (independent)',
      rep: 'ECFP4 + descriptors',
      split: 'same held-out molecules',
      r2: dmpnn.independent_validation.independent.r2_score,
      rmse: dmpnn.independent_validation.independent.rmse,
      n: dmpnn.independent_validation.n_test,
    },
  ].filter(Boolean) as any[];

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      <div className="mb-12">
        <div className="in-label mb-4">module 06 — measured performance</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="What it scores" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p className="mt-5 text-sm max-w-xl leading-relaxed" style={{ color: 'var(--in-dim)' }}>
            Every figure is computed at startup from a held-out split. Nothing on this page is
            written into the source — model version {data.model_version}.
          </p>
        </Rise>
      </div>

      {/* Headline metrics */}
      <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
        {[
          { l: 'R² (random split)', v: m.r2_score, d: 3, accent: true },
          { l: 'RMSE', v: m.rmse, d: 3, u: 'logS' },
          { l: 'MAE', v: m.mae, d: 3, u: 'logS' },
          { l: 'test molecules', v: m.test_samples, d: 0 },
        ].map((s) => (
          <StaggerItem key={s.l}>
            <div className="in-panel in-panel-hover p-5">
              <div className="in-label">{s.l}</div>
              <div
                className="in-num text-3xl font-light mt-2"
                style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}
              >
                <CountUp value={s.v} decimals={s.d} />
                {s.u && <span className="text-[10px] ml-1.5" style={{ color: 'var(--in-faint)' }}>{s.u}</span>}
              </div>
            </div>
          </StaggerItem>
        ))}
      </Stagger>

      {/* Model comparison */}
      <Rise delay={0.15}>
        <div className="in-panel p-6 mb-3 overflow-x-auto">
          <div className="in-label mb-4">model comparison — same held-out molecules</div>
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                {['model', 'representation', 'split', 'R²', 'RMSE', 'n'].map((h) => (
                  <th key={h} className="in-label text-left py-3 px-3 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.model}-${r.split}`} style={{ borderBottom: '1px solid var(--pp-rule-soft)' }}>
                  <td className="py-3 px-3" style={{ color: r.accent ? 'var(--in-signal)' : 'var(--in-text)' }}>
                    {r.model}
                  </td>
                  <td className="py-3 px-3" style={{ color: 'var(--in-dim)' }}>{r.rep}</td>
                  <td className="py-3 px-3" style={{ color: r.caution ? 'var(--in-caution)' : 'var(--in-dim)' }}>
                    {r.split}
                  </td>
                  <td className="py-3 px-3 in-num" style={{ color: r.accent ? 'var(--in-signal)' : 'var(--in-text)' }}>
                    {r.r2}
                  </td>
                  <td className="py-3 px-3 in-num" style={{ color: 'var(--in-dim)' }}>{r.rmse}</td>
                  <td className="py-3 px-3 in-num" style={{ color: 'var(--in-faint)' }}>{r.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {dmpnn?.status === 'trained' && dmpnn.metrics?.solubility && (
            <p className="text-[11px] mt-4 leading-relaxed" style={{ color: 'var(--in-caution)' }}>
              The graph network <strong>matched</strong> rather than beat the tree baseline
              ({dmpnn.metrics.solubility.r2_score} vs {m.r2_score}). That is the expected result
              at this dataset size; it is kept for its calibrated per-molecule sigma, its learned
              latent space, and its multi-task gain on toxicity.
            </p>
          )}
        </div>
      </Rise>

      {/* Parity plot + conformal coverage */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        <Rise delay={0.2}>
          <div className="in-panel p-6 h-full">
            <div className="in-label mb-4">predicted against measured</div>
            <div className="h-[340px]">
              <Plot
                data={[
                  {
                    x: [lo, hi],
                    y: [lo, hi],
                    mode: 'lines',
                    type: 'scatter',
                    name: 'perfect',
                    line: { color: 'rgba(23,23,26,0.2)', dash: 'dot', width: 1 },
                    hoverinfo: 'skip',
                  },
                  {
                    x: actual,
                    y: predicted,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'test molecules',
                    text: parity.map(
                      (p: any) => `<b>${p.compound_id}</b><br>measured ${p.actual}<br>predicted ${p.predicted}<br>90%: [${p.lower_90}, ${p.upper_90}]`
                    ),
                    hoverinfo: 'text',
                    marker: { color: PLOT_COLORS.signal, size: 6, opacity: 0.7 },
                  },
                ] as any}
                layout={plotLayout({ x: 'measured logS', y: 'predicted logS' }) as any}
                config={PLOT_CONFIG as any}
                style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>
        </Rise>

        <Rise delay={0.25}>
          <div className="in-panel p-6 h-full">
            <div className="in-label mb-4">conformal coverage — promised against measured</div>
            <div className="space-y-5 mt-6">
              {Object.entries(m.conformal_test_coverage).map(([level, v]: [string, any]) => {
                const nominal = v.nominal_confidence * 100;
                const got = v.empirical_coverage;
                return (
                  <div key={level}>
                    <div className="flex justify-between mb-2">
                      <span className="in-label">promised {level}</span>
                      <span className="in-num text-[11px]" style={{ color: 'var(--in-signal)' }}>
                        measured {got}%
                      </span>
                    </div>
                    <div className="relative h-1.5 rounded-full" style={{ background: 'var(--pp-rule-soft)' }}>
                      <div
                        className="absolute inset-y-0 left-0 rounded-full"
                        style={{ width: `${got}%`, background: 'var(--in-signal)' }}
                      />
                      <div
                        className="absolute inset-y-[-3px] w-[1px]"
                        style={{ left: `${nominal}%`, background: 'var(--pp-ink)' }}
                        title={`nominal ${nominal}%`}
                      />
                    </div>
                    <div className="in-label mt-1.5">mean width {v.avg_interval_width} logS</div>
                  </div>
                );
              })}
            </div>
            <p className="text-[10px] mt-6 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
              The white tick is the promise; the bar is what was measured on held-out molecules.
              Coverage is marginal across molecules, not a per-molecule probability.
            </p>
          </div>
        </Rise>
      </div>

      {/* Classification + AD */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        {clintox?.status === 'available' && (
          <Rise delay={0.3}>
            <div className="in-panel p-6 h-full">
              <div className="in-label mb-4">classification — ClinTox CT_TOX</div>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { l: 'ROC-AUC', v: clintox.roc_auc },
                  { l: 'PR-AUC', v: clintox.pr_auc, accent: true },
                  { l: 'baseline rate', v: clintox.baseline_pr_auc },
                ].map((s) => (
                  <div key={s.l} className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label">{s.l}</div>
                    <div className="in-num text-xl mt-1.5" style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}>
                      {s.v}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] mt-4 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                PR-AUC must be read against the baseline rate, not against ROC-AUC. CT_TOX marks
                a trial terminated for toxicity, not an intrinsically poisonous molecule.
              </p>
            </div>
          </Rise>
        )}

        {data.applicability_domain?.n_training_molecules && (
          <Rise delay={0.35}>
            <div className="in-panel p-6 h-full">
              <div className="in-label mb-4">applicability domain — derived thresholds</div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { l: 'similarity cut-off', v: data.applicability_domain.similarity_threshold },
                  { l: 'leverage cut-off', v: data.applicability_domain.leverage_threshold },
                  { l: 'OECD reference', v: data.applicability_domain.oecd_leverage_threshold },
                  {
                    l: 'training flagged',
                    v: `${(data.applicability_domain.training_flag_rate * 100).toFixed(1)}%`,
                  },
                ].map((s) => (
                  <div key={s.l} className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label">{s.l}</div>
                    <div className="in-num text-lg mt-1.5">{s.v}</div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] mt-4 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                Percentiles of the training set's own leave-one-out statistics — not literature
                constants.
              </p>
            </div>
          </Rise>
        )}
      </div>

      {/* Provenance + downloads */}
      <Rise delay={0.4}>
        <div className="in-panel p-6">
          <div className="flex flex-wrap items-start justify-between gap-5">
            <div>
              <div className="in-label">dataset provenance</div>
              <div className="in-num text-[11px] mt-2 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                {m.provenance?.dataset} · {m.provenance?.n_used} used ·{' '}
                {m.provenance?.n_duplicates_removed} duplicates removed · hash{' '}
                {m.provenance?.data_hash}
              </div>
              <div className="in-label mt-2">
                train {m.train_samples} · calibration {m.cal_samples} · test {m.test_samples} ·
                campaign pool {m.al_reserved_samples}
              </div>
            </div>
            <div className="flex gap-2">
              {[
                { href: '/api/benchmark/report?format=json', label: 'report · json' },
                { href: '/api/benchmark/report?format=csv', label: 'parity · csv' },
              ].map((d) => (
                <a
                  key={d.href}
                  href={d.href}
                  className="in-btn px-4 py-2.5 inline-flex items-center gap-2"
                  style={{ border: '1px solid var(--in-line-strong)', color: 'var(--in-text)' }}
                >
                  <Download className="w-3 h-3" /> {d.label}
                </a>
              ))}
            </div>
          </div>
        </div>
      </Rise>
    </div>
  );
};
