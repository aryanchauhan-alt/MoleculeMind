/**
 * PredictionModule — module 02.
 *
 * This is the module the whole design system was built for. It drives the
 * shared canvas's uncertainty, so the molecule behind the interface physically
 * blurs and grows a thicker probability shell when the models are unsure, and
 * goes hazy and cold when the applicability domain rejects it.
 *
 * Every value carries its provenance. Properties with no trained model are
 * shown as explicitly unavailable rather than omitted, because a gap the user
 * can see is worth more than a gap they cannot.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Ban, Loader2, Network, TriangleAlert } from 'lucide-react';
import {
  CountUp,
  DUR,
  EASE,
  Magnetic,
  RevealText,
  Rise,
  Stagger,
  StaggerItem,
  Uncertain,
} from './primitives';
import { MoleculeFigure, parseMolBlock, useScene } from './SceneCanvas';

const PRESETS = [
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Warfarin', smiles: 'CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O' },
  { name: 'C36 chain', smiles: 'C'.repeat(36) },
];

const LEVELS = ['80%', '90%', '95%'] as const;
type Level = (typeof LEVELS)[number];

const AD_TONE: Record<string, { color: string; label: string }> = {
  IN_DOMAIN: { color: 'var(--in-signal)', label: 'in domain' },
  BORDERLINE: { color: 'var(--in-caution)', label: 'borderline' },
  OUT_OF_DOMAIN: { color: 'var(--in-alarm)', label: 'out of domain' },
};

interface Props {
  initialSmiles?: string;
}

export const PredictionModule: React.FC<Props> = ({ initialSmiles }) => {
  const scene = useScene();
  const [smiles, setSmiles] = useState(initialSmiles || PRESETS[0].smiles);
  const [result, setResult] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState<Level>('90%');

  const predict = useCallback(
    async (target?: string) => {
      const query = (target ?? smiles).trim();
      if (!query) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: query }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Prediction failed.');
        setResult(json);

        if (json.sdf_3d) {
          const { atoms, bonds } = parseMolBlock(json.sdf_3d);
          scene.setMolecule(atoms, bonds);
        }

        // Drive the canvas from the real calibrated uncertainty. The interval
        // half-width at 90% is the most honest single scalar we have.
        const sol = json.properties?.solubility_logs;
        const margin = sol?.conformal?.intervals?.['90%']?.margin_error ?? 0;
        scene.setUncertainty(Math.min(margin / 2.5, 1));
        scene.setOutOfDomain(json.applicability_domain?.status === 'OUT_OF_DOMAIN');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Prediction failed.');
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [smiles, scene]
  );

  useEffect(() => {
    scene.setMode('prediction');
    predict(initialSmiles || PRESETS[0].smiles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const props = result?.properties;
  const sol = props?.solubility_logs;
  const ad = result?.applicability_domain;
  const interval = sol?.conformal?.intervals?.[level];
  const sigma = interval?.margin_error ?? 0;

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      <div className="mb-12">
        <div className="in-label mb-4">module 02 — property prediction</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="How sure are we?" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p className="mt-5 text-sm max-w-xl leading-relaxed" style={{ color: 'var(--in-dim)' }}>
            Every prediction arrives with a calibrated interval. The figure and the number
            below blur in proportion to that uncertainty — when the models are unsure, the
            result is literally harder to read.
          </p>
        </Rise>
      </div>

      {/* Input */}
      <Rise delay={0.2} className="mb-10">
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && predict()}
            spellCheck={false}
            placeholder="Enter a SMILES string"
            className="in-input flex-1 px-5 py-4 text-sm"
            style={{ color: 'var(--in-text)' }}
          />
          <Magnetic strength={4}>
            <button
              onClick={() => predict()}
              disabled={loading}
              className="in-btn px-7 py-4 w-full sm:w-auto cursor-pointer disabled:opacity-40 flex items-center justify-center gap-2"
              style={{ background: 'var(--in-signal)', color: '#FFFFFF', fontWeight: 600 }}
            >
              {loading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              {loading ? 'computing' : 'predict'}
            </button>
          </Magnetic>
        </div>
        <div className="flex flex-wrap gap-2 mt-4">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => {
                setSmiles(p.smiles);
                predict(p.smiles);
              }}
              className="in-btn px-3.5 py-1.5 cursor-pointer"
              style={{
                border: '1px solid var(--in-line)',
                color: 'var(--in-dim)',
                background: 'var(--pp-sunken)',
              }}
            >
              {p.name}
            </button>
          ))}
        </div>
      </Rise>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="in-panel p-5 mb-8 flex gap-3"
            style={{ borderColor: 'rgba(179,38,30,0.3)' }}
          >
            <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" style={{ color: 'var(--in-alarm)' }} />
            <p className="text-xs" style={{ color: 'var(--in-dim)' }}>{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {sol && (
        <>
          {/* --- The headline: value, interval, and the blur that encodes it --- */}
          <Rise delay={0.1}>
            <div className="in-panel p-7 sm:p-9 mb-3">
              <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-8">
                <div>
                  <div className="in-label">aqueous solubility — logS</div>
                  <Uncertain sigma={sigma} scale={2.5} outOfDomain={ad?.status === 'OUT_OF_DOMAIN'}>
                    <div
                      className="in-num text-[14vw] sm:text-[6vw] font-light leading-none mt-3"
                      style={{ color: 'var(--in-signal)' }}
                    >
                      <CountUp value={sol.value} decimals={3} />
                    </div>
                  </Uncertain>
                  <div className="in-label mt-3">{sol.classification} · {sol.unit}</div>
                </div>

                {/* The molecule, blurred by the same uncertainty as the number */}
                <MoleculeFigure height={200} className="w-full lg:w-[280px] shrink-0" />

                {/* Confidence selector */}
                <div className="flex flex-col gap-3">
                  <div className="flex gap-1.5">
                    {LEVELS.map((l) => (
                      <button
                        key={l}
                        onClick={() => setLevel(l)}
                        className="in-btn px-3.5 py-1.5 cursor-pointer"
                        style={{
                          background: l === level ? 'var(--in-signal)' : 'var(--pp-sunken)',
                          color: l === level ? '#FFFFFF' : 'var(--in-dim)',
                          border: `1px solid ${l === level ? 'var(--in-signal)' : 'var(--in-line)'}`,
                          fontWeight: l === level ? 600 : 400,
                        }}
                      >
                        {l}
                      </button>
                    ))}
                  </div>
                  <div className="text-right">
                    <div className="in-label">conformal interval</div>
                    <div className="in-num text-xl mt-1.5">
                      [{interval?.lower_bound}, {interval?.upper_bound}]
                    </div>
                    <div className="in-label mt-1">width {interval?.interval_width} logS</div>
                  </div>
                </div>
              </div>

              {/* The interval as a physical bar */}
              <div className="mt-8">
                <div className="relative h-10">
                  <div
                    className="absolute inset-y-[18px] left-0 right-0"
                    style={{ background: 'var(--in-line)' }}
                  />
                  <motion.div
                    className="absolute inset-y-[14px] rounded-full"
                    style={{ background: 'var(--in-signal-dim)', border: '1px solid rgba(76,224,224,0.3)' }}
                    initial={false}
                    animate={{ left: '12%', right: '12%' }}
                    transition={{ duration: DUR.base, ease: EASE }}
                  />
                  <motion.div
                    className="absolute w-[2px] inset-y-[8px] left-1/2"
                    style={{ background: 'var(--in-signal)' }}
                    layout
                  />
                </div>
                <div className="flex justify-between in-label">
                  <span>{interval?.lower_bound}</span>
                  <span style={{ color: 'var(--in-signal)' }}>{sol.value}</span>
                  <span>{interval?.upper_bound}</span>
                </div>
              </div>

              <p className="text-[11px] mt-6 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
                {level} of such intervals contain the true value, averaged over many molecules.
                This is <em>not</em> a {level} chance that this particular molecule falls inside
                its own interval — conformal coverage is marginal, not per-molecule.
              </p>
            </div>
          </Rise>

          {/* --- Applicability domain --- */}
          {ad && (
            <Rise delay={0.2}>
              <div
                className="in-panel p-7 mb-3"
                style={{ borderColor: `color-mix(in srgb, ${AD_TONE[ad.status]?.color} 25%, transparent)` }}
              >
                <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
                  <div>
                    <div className="in-label">applicability domain</div>
                    <div
                      className="in-display text-2xl mt-2"
                      style={{ color: AD_TONE[ad.status]?.color }}
                    >
                      {AD_TONE[ad.status]?.label ?? ad.status}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="in-label">domain index</div>
                    <div className="in-num text-2xl mt-1.5">
                      <CountUp value={ad.domain_index} decimals={3} />
                    </div>
                  </div>
                </div>

                <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    { l: 'max similarity', v: ad.metrics.max_tanimoto_similarity, t: ad.metrics.similarity_threshold },
                    { l: 'latent distance', v: ad.metrics.latent_distance, t: ad.metrics.latent_threshold },
                    { l: 'leverage', v: ad.metrics.leverage, t: ad.metrics.leverage_threshold },
                    { l: 'nearest analogue', v: null, t: null },
                  ].map((m) => (
                    <StaggerItem key={m.l}>
                      <div className="p-4 rounded-xl h-full" style={{ background: 'var(--pp-sunken)' }}>
                        <div className="in-label">{m.l}</div>
                        {m.v !== null && m.v !== undefined ? (
                          <>
                            <div className="in-num text-lg mt-1.5">{m.v}</div>
                            <div className="in-label mt-1">threshold {m.t}</div>
                          </>
                        ) : m.l === 'nearest analogue' ? (
                          <div className="in-num text-[10px] mt-1.5 break-all" style={{ color: 'var(--in-dim)' }}>
                            {ad.nearest_analog?.smiles?.slice(0, 28)}
                            <div className="in-label mt-1">
                              measured {ad.nearest_analog?.measured_value}
                            </div>
                          </div>
                        ) : (
                          <div className="in-num text-sm mt-1.5" style={{ color: 'var(--in-faint)' }}>—</div>
                        )}
                      </div>
                    </StaggerItem>
                  ))}
                </Stagger>

                <p className="text-[11px] mt-5 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                  {ad.explanation}
                </p>
                <p className="text-[10px] mt-2 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
                  {ad.index_caveat}
                </p>
              </div>
            </Rise>
          )}

          {/* --- All properties with provenance --- */}
          <Rise delay={0.3}>
            <div className="in-label mb-4">property profile</div>
            <Stagger className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-3">
              {Object.entries(props).map(([key, p]: [string, any]) => (
                <StaggerItem key={key}>
                  {p.status === 'available' ? (
                    <div className="in-panel in-panel-hover p-5 h-full">
                      <div className="in-label">{p.property_name}</div>
                      <div className="in-num text-2xl font-light mt-2">
                        {typeof p.value === 'number' ? <CountUp value={p.value} decimals={3} /> : p.value}
                      </div>
                      <div className="flex items-center gap-1.5 mt-3">
                        <span
                          className="in-btn px-2 py-0.5"
                          style={{
                            background: p.provenance === 'model' ? 'var(--in-signal-dim)' : 'var(--pp-sunken)',
                            color: p.provenance === 'model' ? 'var(--in-signal)' : 'var(--in-dim)',
                            border: '1px solid var(--in-line)',
                            fontSize: 9,
                          }}
                          title={p.model}
                        >
                          {p.provenance}
                        </span>
                        <span className="in-label">{p.unit}</span>
                      </div>
                      {p.caveat && (
                        <p className="text-[10px] mt-3 leading-relaxed" style={{ color: 'var(--in-caution)' }}>
                          {p.caveat}
                        </p>
                      )}
                      {p.memorisation_warning && (
                        <p className="text-[10px] mt-2 leading-relaxed" style={{ color: 'var(--in-alarm)' }}>
                          {p.memorisation_warning}
                        </p>
                      )}
                    </div>
                  ) : (
                    <div
                      className="p-5 h-full rounded-[20px]"
                      style={{ border: '1px dashed var(--in-line-strong)', background: 'var(--pp-sunken)' }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="in-label">{p.property_name}</div>
                        <Ban className="w-3.5 h-3.5 shrink-0" style={{ color: 'var(--in-faint)' }} />
                      </div>
                      <div className="in-display text-sm mt-2" style={{ color: 'var(--in-faint)' }}>
                        unavailable
                      </div>
                      <p className="text-[10px] mt-2.5 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
                        {p.reason}
                      </p>
                    </div>
                  )}
                </StaggerItem>
              ))}
            </Stagger>
          </Rise>

          {/* --- Independent cross-check --- */}
          {result.dmpnn?.solubility_logs && (
            <Rise delay={0.4}>
              <div className="in-panel p-7">
                <div className="flex items-center gap-2 mb-2">
                  <Network className="w-4 h-4" style={{ color: 'var(--in-signal)' }} />
                  <div className="in-display text-sm">Independent cross-check</div>
                </div>
                <p className="text-[11px] mb-6 leading-relaxed max-w-3xl" style={{ color: 'var(--in-dim)' }}>
                  A graph neural network on a different representation. Agreement is weak
                  evidence the prediction is not an artefact of how the molecule was encoded —
                  not proof it is correct.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {[
                    {
                      l: 'graph network',
                      v: result.dmpnn.solubility_logs.value,
                      sub: `σ = ${result.dmpnn.solubility_logs.sigma} · ${result.dmpnn.n_members} members`,
                      accent: true,
                    },
                    {
                      l: 'tree baseline',
                      v: result.dmpnn.agreement_with_baseline.baseline_value,
                      sub: 'ExtraTrees on fingerprints',
                    },
                    {
                      l: 'difference',
                      v: result.dmpnn.agreement_with_baseline.absolute_difference,
                      sub: result.dmpnn.agreement_with_baseline.exceeds_dmpnn_sigma
                        ? 'exceeds σ — representation-sensitive'
                        : 'within σ — models agree',
                      warn: result.dmpnn.agreement_with_baseline.exceeds_dmpnn_sigma,
                    },
                  ].map((c) => (
                    <div
                      key={c.l}
                      className="p-5 rounded-xl"
                      style={{
                        background: 'var(--pp-sunken)',
                        border: `1px solid ${c.warn ? 'rgba(180,83,15,0.25)' : 'var(--in-line)'}`,
                      }}
                    >
                      <div className="in-label">{c.l}</div>
                      <div
                        className="in-num text-xl font-light mt-2"
                        style={{
                          color: c.warn
                            ? 'var(--in-caution)'
                            : c.accent
                            ? 'var(--in-signal)'
                            : 'var(--in-text)',
                        }}
                      >
                        {c.v}
                      </div>
                      <div className="in-label mt-1.5">{c.sub}</div>
                    </div>
                  ))}
                </div>

                {/* Variance decomposition */}
                <div className="grid grid-cols-2 gap-3 mt-3">
                  <div className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label">aleatoric variance</div>
                    <div className="in-num text-sm mt-1.5">
                      {result.dmpnn.solubility_logs.aleatoric_variance}
                      <span className="ml-2" style={{ color: 'var(--in-faint)' }}>
                        noise in the data
                      </span>
                    </div>
                  </div>
                  <div className="p-4 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label">epistemic variance</div>
                    <div className="in-num text-sm mt-1.5">
                      {result.dmpnn.solubility_logs.epistemic_variance}
                      <span className="ml-2" style={{ color: 'var(--in-faint)' }}>
                        model disagreement
                      </span>
                    </div>
                  </div>
                </div>

                {/* Conformal prediction sets */}
                {result.dmpnn.toxicity?.conformal_sets && (
                  <div className="mt-3 p-5 rounded-xl" style={{ background: 'var(--pp-sunken)' }}>
                    <div className="in-label mb-3">conformal prediction set — toxicity, 90%</div>
                    <div className="flex flex-wrap gap-2">
                      {(['lac', 'aps_randomized'] as const).map((m) => {
                        const set = result.dmpnn.toxicity.conformal_sets[m];
                        if (!set) return null;
                        return (
                          <span
                            key={m}
                            className="in-btn px-3 py-1.5"
                            style={{ border: '1px solid var(--in-line)', color: 'var(--in-text)' }}
                            title={set.method}
                          >
                            {m === 'lac' ? 'LAC' : 'APS'} ·{' '}
                            {set.labels.length ? set.labels.join(' | ') : 'empty set'}
                          </span>
                        );
                      })}
                    </div>
                    <p className="text-[10px] mt-3 leading-relaxed" style={{ color: 'var(--in-faint)' }}>
                      A set, not a single label. Empty means no label was plausible; both means
                      the model cannot discriminate.
                    </p>
                  </div>
                )}
              </div>
            </Rise>
          )}
        </>
      )}
    </div>
  );
};
