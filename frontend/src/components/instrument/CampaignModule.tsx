/**
 * CampaignModule — module 05, active learning.
 *
 * The honesty requirement here is structural, not cosmetic: the candidate pool
 * carries no measured value until its assay has actually been run, because the
 * backend keeps the oracle server-side. The interface therefore shows a dash,
 * not a hidden number, and the simulated nature of the assay is stated in the
 * header rather than buried in a tooltip.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FlaskConical, Loader2, RotateCcw } from 'lucide-react';
import { CountUp, DUR, EASE, Magnetic, RevealText, Rise, Stagger, StaggerItem } from './primitives';
import { parseMolBlock, useScene } from './SceneCanvas';

const STRATEGIES = [
  { id: 'ucb', label: 'UCB', blurb: 'mu + sqrt(beta)·sigma — balanced' },
  { id: 'ei', label: 'Expected improvement', blurb: 'expected gain over the best so far' },
  { id: 'uncertainty', label: 'Uncertainty', blurb: 'sigma alone — pure exploration' },
  { id: 'greedy', label: 'Greedy', blurb: 'mu alone — pure exploitation' },
];

export const CampaignModule: React.FC<{ onSelect?: (smiles: string) => void }> = ({ onSelect }) => {
  const scene = useScene();
  const [data, setData] = useState<any | null>(null);
  const [strategy, setStrategy] = useState('ucb');
  const [beta, setBeta] = useState(2);
  const [loading, setLoading] = useState(true);
  const [assaying, setAssaying] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (s = strategy, b = beta) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/active-learning/pool?strategy=${s}&beta=${b}`);
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Could not load the campaign.');
        setData(json);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not load the campaign.');
      } finally {
        setLoading(false);
      }
    },
    [strategy, beta]
  );

  useEffect(() => {
    scene.setMode('active_learning');
    scene.setUncertainty(0.35);
    scene.setOutOfDomain(false);
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const t = window.setTimeout(() => load(strategy, beta), 250);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy, beta]);

  const runAssay = async (id: string, smiles: string) => {
    setAssaying(id);
    try {
      const res = await fetch('/api/active-learning/assay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_ids: [id] }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Assay failed.');
      await load();
      // Show the assayed molecule on the canvas.
      const a = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles }),
      }).then((r) => (r.ok ? r.json() : null));
      if (a?.sdf_3d) {
        const { atoms, bonds } = parseMolBlock(a.sdf_3d);
        scene.setMolecule(atoms, bonds);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Assay failed.');
    } finally {
      setAssaying(null);
    }
  };

  const reset = async () => {
    await fetch('/api/active-learning/reset', { method: 'POST' });
    load();
  };

  const candidates = data?.candidates ?? [];
  const unlabelled = candidates
    .filter((c: any) => !c.assayed)
    .sort((a: any, b: any) => (b.acquisition_score ?? 0) - (a.acquisition_score ?? 0));
  const labelled = candidates.filter((c: any) => c.assayed);
  const diag = data?.acquisition_diagnostics;

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      <div className="mb-12">
        <div className="in-label mb-4">module 05 — active learning campaign</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="What next?" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p className="mt-5 text-sm max-w-2xl leading-relaxed" style={{ color: 'var(--in-dim)' }}>
            A Gaussian process scores every unmeasured candidate. Running an assay here is a
            <strong style={{ color: 'var(--in-caution)' }}> simulation</strong>: it reveals a
            measurement withheld from the model, taken from the Delaney dataset. No laboratory
            experiment is performed.
          </p>
        </Rise>
      </div>

      {/* Strategy */}
      <Rise delay={0.15}>
        <div className="in-panel p-6 mb-3">
          <div className="in-label mb-3">acquisition strategy</div>
          <div className="flex flex-wrap gap-2 mb-5">
            {STRATEGIES.map((s) => (
              <button
                key={s.id}
                onClick={() => setStrategy(s.id)}
                title={s.blurb}
                className="in-btn px-4 py-2 cursor-pointer"
                style={{
                  background: strategy === s.id ? 'var(--in-signal)' : 'rgba(0,0,0,0.3)',
                  color: strategy === s.id ? '#04161A' : 'var(--in-dim)',
                  border: `1px solid ${strategy === s.id ? 'var(--in-signal)' : 'var(--in-line)'}`,
                  fontWeight: strategy === s.id ? 600 : 400,
                }}
              >
                {s.label}
              </button>
            ))}
          </div>

          {strategy === 'ucb' && (
            <div className="max-w-sm">
              <div className="flex justify-between mb-2">
                <span className="in-label">exploration β</span>
                <span className="in-num text-[11px]" style={{ color: 'var(--in-signal)' }}>
                  {beta.toFixed(1)}
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={10}
                step={0.1}
                value={beta}
                onChange={(e) => setBeta(parseFloat(e.target.value))}
                className="w-full cursor-pointer"
                style={{ accentColor: 'var(--in-signal)' }}
              />
            </div>
          )}

          <p className="text-[11px] mt-4" style={{ color: 'var(--in-faint)' }}>
            {STRATEGIES.find((s) => s.id === strategy)?.blurb}
          </p>
        </div>
      </Rise>

      {/* Campaign state */}
      {data && (
        <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
          {[
            { l: 'round', v: data.iteration, d: 0 },
            { l: 'measured', v: labelled.length, d: 0, accent: true },
            { l: 'remaining', v: unlabelled.length, d: 0 },
            {
              l: 'best measured logS',
              v: data.history?.[data.history.length - 1]?.best_revealed_logs ?? 0,
              d: 2,
            },
          ].map((s) => (
            <StaggerItem key={s.l}>
              <div className="in-panel in-panel-hover p-5">
                <div className="in-label">{s.l}</div>
                <div
                  className="in-num text-3xl font-light mt-2"
                  style={{ color: s.accent ? 'var(--in-signal)' : 'var(--in-text)' }}
                >
                  <CountUp value={Number(s.v)} decimals={s.d} />
                </div>
              </div>
            </StaggerItem>
          ))}
        </Stagger>
      )}

      {/* Acquisition health — the regression guard made visible */}
      {diag && (
        <Rise delay={0.2}>
          <div
            className="in-panel p-5 mb-3 flex flex-wrap items-center gap-x-8 gap-y-3"
            style={{ borderColor: diag.degenerate ? 'rgba(240,113,91,0.3)' : 'var(--in-line)' }}
          >
            <div>
              <div className="in-label">acquisition spread</div>
              <div className="in-num text-sm mt-1">
                {diag.n_unique} distinct scores across {diag.n} candidates
              </div>
            </div>
            <div>
              <div className="in-label">range</div>
              <div className="in-num text-sm mt-1">
                {diag.min?.toFixed(3)} … {diag.max?.toFixed(3)}
              </div>
            </div>
            <div
              className="in-btn px-3 py-1.5"
              style={{
                background: diag.degenerate ? 'rgba(240,113,91,0.1)' : 'var(--in-signal-dim)',
                color: diag.degenerate ? 'var(--in-alarm)' : 'var(--in-signal)',
                border: '1px solid var(--in-line)',
              }}
            >
              {diag.degenerate ? 'degenerate — ranking meaningless' : 'discriminative'}
            </div>
            <p className="text-[10px] flex-1 min-w-[260px]" style={{ color: 'var(--in-faint)' }}>
              This readout exists because an earlier version gave all 24 candidates the same
              score, which made the ranking meaningless while looking like it worked.
            </p>
          </div>
        </Rise>
      )}

      {error && (
        <div className="in-panel p-5 mb-3" style={{ borderColor: 'rgba(240,113,91,0.25)' }}>
          <p className="text-xs" style={{ color: 'var(--in-dim)' }}>{error}</p>
        </div>
      )}

      {loading && !data && (
        <div className="in-panel p-16 flex flex-col items-center gap-4">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--in-signal)' }} />
          <span className="in-label">fitting the surrogate</span>
        </div>
      )}

      {/* Queue */}
      {data && (
        <Rise delay={0.25}>
          <div className="in-panel p-6 mb-3">
            <div className="flex items-center justify-between mb-5">
              <div className="in-label">recommended next experiments</div>
              <Magnetic strength={3}>
                <button
                  onClick={reset}
                  className="in-btn px-4 py-2 cursor-pointer flex items-center gap-2"
                  style={{ border: '1px solid var(--in-line)', color: 'var(--in-dim)' }}
                >
                  <RotateCcw className="w-3 h-3" /> reset campaign
                </button>
              </Magnetic>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--in-line)' }}>
                    {['#', 'compound', 'acquisition', 'predicted', 'sigma', 'QED', ''].map((h) => (
                      <th key={h} className="in-label text-left py-3 px-3 whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence initial={false}>
                    {unlabelled.slice(0, 10).map((c: any, i: number) => (
                      <motion.tr
                        key={c.id}
                        layout
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: DUR.fast, ease: EASE }}
                        style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}
                      >
                        <td className="py-3 px-3 in-num" style={{ color: i === 0 ? 'var(--in-signal)' : 'var(--in-faint)' }}>
                          {String(i + 1).padStart(2, '0')}
                        </td>
                        <td className="py-3 px-3">
                          <button
                            onClick={() => onSelect?.(c.smiles)}
                            className="hover:underline cursor-pointer text-left"
                          >
                            {c.name}
                          </button>
                          {i === 0 && (
                            <span className="in-label ml-2" style={{ color: 'var(--in-signal)' }}>
                              top pick
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-3 in-num" style={{ color: 'var(--in-signal)' }}>
                          {c.acquisition_score ?? '—'}
                        </td>
                        <td className="py-3 px-3 in-num" style={{ color: 'var(--in-dim)' }}>
                          {c.predicted_value ?? '—'}
                        </td>
                        <td className="py-3 px-3 in-num" style={{ color: 'var(--in-caution)' }}>
                          ±{c.uncertainty_sigma ?? '—'}
                        </td>
                        <td className="py-3 px-3 in-num" style={{ color: 'var(--in-dim)' }}>{c.qed}</td>
                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={() => runAssay(c.id, c.smiles)}
                            disabled={assaying !== null}
                            className="in-btn px-3.5 py-1.5 cursor-pointer disabled:opacity-40 inline-flex items-center gap-1.5"
                            style={{
                              background: i === 0 ? 'var(--in-signal)' : 'rgba(0,0,0,0.3)',
                              color: i === 0 ? '#04161A' : 'var(--in-dim)',
                              border: `1px solid ${i === 0 ? 'var(--in-signal)' : 'var(--in-line)'}`,
                              fontWeight: i === 0 ? 600 : 400,
                            }}
                          >
                            {assaying === c.id ? (
                              <Loader2 className="w-3 h-3 animate-spin" />
                            ) : (
                              <FlaskConical className="w-3 h-3" />
                            )}
                            assay
                          </button>
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>

            <p className="text-[10px] mt-4 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
              Unmeasured candidates show no value because the backend does not send one — the
              oracle stays server-side until an assay is executed.
            </p>
          </div>
        </Rise>
      )}

      {/* Measured so far */}
      {labelled.length > 0 && (
        <Rise delay={0.3}>
          <div className="in-panel p-6">
            <div className="in-label mb-4">measured ({labelled.length})</div>
            <div className="flex flex-wrap gap-2">
              {labelled.map((c: any) => (
                <div
                  key={c.id}
                  className="px-3.5 py-2 rounded-xl"
                  style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--in-line)' }}
                >
                  <div className="text-[11px]">{c.name}</div>
                  <div className="in-num text-[10px] mt-0.5" style={{ color: 'var(--in-signal)' }}>
                    logS {c.measured_value} · round {c.assay_round}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Rise>
      )}
    </div>
  );
};
