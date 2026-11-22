import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { MoleculeFigure, parseMolBlock, useScene } from './instrument/SceneCanvas';
import { Rise, Stagger, StaggerItem, Uncertain } from './instrument/primitives';

/**
 * The overview page, on paper.
 *
 * Two rules govern everything here.
 *
 * 1. NOTHING ON THIS PAGE IS A WRITTEN-IN NUMBER. The previous version
 *    shipped a `PRESET_SHOWCASE` table of hardcoded logS values and
 *    conformal intervals, plus a "< 38 ms" latency claim. Those are
 *    fabricated results dressed as model output: they would not move if
 *    the model were retrained, or deleted. Every figure below is fetched
 *    from the running backend, or absent.
 *
 * 2. No backdrop blur, no glow, no particle field, no marquee. The module
 *    views were rebuilt on the PAPER system; this page had been left on
 *    the old dark theme, which is why the site still looked like two
 *    different products stitched together.
 */

type TabId =
  | 'analyzer'
  | 'prediction'
  | 'pareto'
  | 'discovery'
  | 'active_learning'
  | 'benchmark';

interface LandingPageProps {
  onLaunchApp: (tab?: TabId) => void;
  onOpenGlossary: () => void;
  backendHealthy: boolean;
}

/** Structures only. Whatever the models say about them is fetched live. */
const SHOWCASE = [
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Ibuprofen', smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' },
  { name: 'Diclofenac', smiles: 'O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl' },
  { name: 'Imatinib core', smiles: 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1' },
  { name: 'C36 alkane', smiles: 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC' },
];

const DOMAIN_COLOR: Record<string, string> = {
  IN_DOMAIN: 'var(--pp-accent)',
  BORDERLINE: 'var(--pp-caution)',
  OUT_OF_DOMAIN: 'var(--pp-alarm)',
};

export const LandingPage: React.FC<LandingPageProps> = ({
  onLaunchApp,
  onOpenGlossary,
  backendHealthy,
}) => {
  const scene = useScene();

  // --- benchmark figures, fetched -------------------------------------
  const [metrics, setMetrics] = useState<any | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/benchmark')
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (!cancelled && j) setMetrics(j.metrics);
      })
      .catch(() => {
        /* the page renders without metrics when the backend is down */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const fmt = (v: any, digits = 3) =>
    typeof v === 'number' ? v.toFixed(digits) : '—';
  const coverage90 = metrics?.conformal_test_coverage?.['90%']?.empirical_coverage;

  // --- live showcase ---------------------------------------------------
  const [picked, setPicked] = useState(SHOWCASE[0]);
  const [demo, setDemo] = useState<any | null>(null);
  const [demoError, setDemoError] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);

  const runDemo = useCallback(
    async (smiles: string) => {
      setDemoLoading(true);
      setDemoError(null);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Prediction failed.');
        setDemo(json);

        if (json.sdf_3d) {
          const { atoms, bonds } = parseMolBlock(json.sdf_3d);
          scene.setMolecule(atoms, bonds);
        }
        const margin =
          json.properties?.solubility_logs?.conformal?.intervals?.['90%']
            ?.margin_error ?? 0;
        scene.setUncertainty(Math.min(margin / 2.5, 1));
        scene.setOutOfDomain(
          json.applicability_domain?.status === 'OUT_OF_DOMAIN'
        );
      } catch (e) {
        setDemo(null);
        setDemoError(e instanceof Error ? e.message : 'Prediction failed.');
      } finally {
        setDemoLoading(false);
      }
    },
    [scene]
  );

  useEffect(() => {
    scene.setMode('idle');
    runDemo(picked.smiles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [picked.smiles]);

  const sol = demo?.properties?.solubility_logs;
  const iv90 = sol?.conformal?.intervals?.['90%'];
  const domain = demo?.applicability_domain;

  const MODULES: {
    id: TabId;
    num: string;
    title: string;
    what: string;
    how: string;
  }[] = [
    {
      id: 'analyzer',
      num: '01',
      title: 'Analyzer',
      what: 'Reads a SMILES string, draws the structure and builds a 3D conformer.',
      how: 'RDKit parsing and canonicalisation, ETKDGv3 embedding, MMFF94 minimisation, Lipinski / Veber / QED descriptors.',
    },
    {
      id: 'prediction',
      num: '02',
      title: 'Prediction',
      what: 'Predicts aqueous solubility and returns a calibrated interval, not a bare number.',
      how: 'ExtraTrees on 1,024-bit Morgan fingerprints plus 8 descriptors, trained on the Delaney ESOL measurements.',
    },
    {
      id: 'prediction',
      num: '03',
      title: 'Uncertainty',
      what: 'Turns the model spread into an interval with a finite-sample coverage guarantee.',
      how: 'Locally-adaptive split conformal prediction, calibrated on a held-out partition at 80 / 90 / 95%.',
    },
    {
      id: 'prediction',
      num: '04',
      title: 'Applicability domain',
      what: 'Says when the molecule is too unlike the training set for the prediction to mean anything.',
      how: 'Tanimoto distance to the training set and PCA leverage against the OECD h* cutoff, combined as a two-condition rule.',
    },
    {
      id: 'pareto',
      num: '05',
      title: 'Pareto',
      what: 'Finds the genuine trade-offs between competing objectives, without inventing weights.',
      how: 'Fast non-dominated sorting, with the frontier recomputed under sampled uncertainty rather than on point estimates.',
    },
    {
      id: 'active_learning',
      num: '06',
      title: 'Active learning',
      what: 'Chooses which compounds are worth the next round of experiments.',
      how: 'A BoTorch SingleTaskGP surrogate with UCB, Expected Improvement and uncertainty sampling, scored over a hidden pool.',
    },
  ];

  const navLink =
    'hover:opacity-100 opacity-65 transition-opacity cursor-pointer';

  return (
    <div className="min-h-screen flex flex-col">
      {/* ---------------------------------------------------------- nav */}
      <header
        className="sticky top-0 z-40"
        style={{
          background: 'var(--pp-paper)',
          borderBottom: '1px solid var(--pp-rule)',
        }}
      >
        <div className="max-w-[1400px] mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-6">
          <div className="flex items-baseline gap-3">
            <span className="in-display text-[15px]">MoleculeMind</span>
            <span className="in-label hidden sm:inline">
              uncertainty-aware discovery
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-7 in-label">
            <a href="#problem" className={navLink}>
              problem
            </a>
            <a href="#live" className={navLink}>
              live
            </a>
            <a href="#modules" className={navLink}>
              modules
            </a>
            <a href="#validation" className={navLink}>
              validation
            </a>
          </nav>

          <div className="flex items-center gap-3">
            <span
              className="in-label hidden sm:flex items-center gap-2"
              title={
                backendHealthy
                  ? 'Backend reachable'
                  : 'Backend unreachable — figures below will be blank'
              }
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: backendHealthy
                    ? 'var(--pp-accent)'
                    : 'var(--pp-caution)',
                }}
              />
              {backendHealthy ? 'online' : 'offline'}
            </span>
            <button onClick={() => onLaunchApp('analyzer')} className="in-btn">
              Open workbench
            </button>
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------- hero */}
      <section className="max-w-[1400px] w-full mx-auto px-5 sm:px-8 pt-16 sm:pt-24 pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-end">
          <div className="lg:col-span-7">
            <Rise>
              <div className="in-label mb-6">
                an honest instrument, not an oracle
              </div>
            </Rise>

            <motion.h1
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              className="in-display-xl text-[13vw] sm:text-[8.5vw] lg:text-[6.4rem]"
            >
              A prediction
              <br />
              you can
              <br />
              <span style={{ color: 'var(--pp-accent)' }}>argue with.</span>
            </motion.h1>

            <Rise delay={0.15}>
              <p
                className="mt-8 max-w-xl text-[15px] leading-relaxed"
                style={{ color: 'var(--pp-muted)' }}
              >
                Most molecular models hand a chemist a single number and no way
                to tell a good guess from a bad one. This platform refuses to
                do that. Every prediction arrives with a calibrated interval, a
                statement about whether the molecule is even inside the
                training set's chemical space, and the trade-offs it forces
                against every other objective.
              </p>
            </Rise>

            <Rise delay={0.25}>
              <div className="mt-9 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => onLaunchApp('analyzer')}
                  className="in-btn"
                  style={{
                    background: 'var(--pp-ink)',
                    color: 'var(--pp-paper)',
                    borderColor: 'var(--pp-ink)',
                  }}
                >
                  Open the workbench
                </button>
                <button onClick={onOpenGlossary} className="in-btn">
                  Methodology
                </button>
              </div>
            </Rise>
          </div>

          <div className="lg:col-span-5">
            <MoleculeFigure
              height={320}
              caption={
                demo
                  ? `${picked.name} · MMFF94-minimised conformer · haze tracks the calibrated uncertainty`
                  : 'awaiting backend'
              }
            />
          </div>
        </div>

        {/* real figures, or nothing */}
        <Stagger className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-px" delay={0.3}>
          {[
            {
              k: 'test R² (ESOL)',
              v: fmt(metrics?.r2_score),
              n: 'held-out split',
            },
            {
              k: 'empirical coverage',
              v:
                typeof coverage90 === 'number'
                  ? `${coverage90.toFixed(1)}%`
                  : '—',
              n: 'at 90% nominal',
            },
            {
              k: 'RMSE',
              v: fmt(metrics?.rmse),
              n: 'log(mol/L)',
            },
            {
              k: 'test compounds',
              v:
                typeof metrics?.test_samples === 'number'
                  ? String(metrics.test_samples)
                  : '—',
              n: 'never seen in training',
            },
          ].map((s) => (
            <StaggerItem key={s.k}>
              <div
                className="py-6 pr-6"
                style={{ borderTop: '1px solid var(--pp-rule)' }}
              >
                <div className="in-num text-3xl sm:text-4xl font-light">
                  {s.v}
                </div>
                <div className="in-label mt-3">{s.k}</div>
                <div
                  className="text-[11px] mt-1"
                  style={{ color: 'var(--pp-faint)' }}
                >
                  {s.n}
                </div>
              </div>
            </StaggerItem>
          ))}
        </Stagger>
      </section>

      {/* ----------------------------------------------------- problem */}
      <section
        id="problem"
        className="max-w-[1400px] w-full mx-auto px-5 sm:px-8 py-20 sm:py-28"
      >
        <Rise>
          <div className="in-label">the problem</div>
          <h2 className="in-display text-3xl sm:text-5xl mt-4 max-w-3xl">
            Three ways a confident number misleads a chemist.
          </h2>
        </Rise>

        <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-10 md:gap-8">
          {[
            {
              n: '01',
              t: 'A point estimate hides its own error',
              b: 'A model that says −2.5 logS does not say whether the laboratory value is within ±0.3 or ±3.0. The first is worth synthesising on; the second is a coin flip.',
              s: 'answered by split conformal calibration',
            },
            {
              n: '02',
              t: 'Models extrapolate silently',
              b: 'Show a tree ensemble a scaffold unlike anything it was trained on and it still returns a number, with no outward sign that it is guessing far outside its evidence.',
              s: 'answered by an OECD-style applicability domain',
            },
            {
              n: '03',
              t: 'Objectives genuinely conflict',
              b: 'What makes a molecule dissolve better often makes it less drug-like, and reducing toxicity can cost both. There is no single best compound, only a set of honest trade-offs.',
              s: 'answered by non-dominated Pareto sorting',
            },
          ].map((c) => (
            <div key={c.n} style={{ borderTop: '1px solid var(--pp-rule)' }}>
              <div className="in-num text-sm pt-5" style={{ color: 'var(--pp-faint)' }}>
                {c.n} / 03
              </div>
              <h3 className="in-display text-xl mt-4">{c.t}</h3>
              <p
                className="text-sm leading-relaxed mt-3"
                style={{ color: 'var(--pp-muted)' }}
              >
                {c.b}
              </p>
              <div
                className="in-label mt-5"
                style={{ color: 'var(--pp-accent)' }}
              >
                {c.s}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* -------------------------------------------------------- live */}
      <section
        id="live"
        style={{
          background: 'var(--pp-sunken)',
          borderTop: '1px solid var(--pp-rule)',
          borderBottom: '1px solid var(--pp-rule)',
        }}
      >
        <div className="max-w-[1400px] w-full mx-auto px-5 sm:px-8 py-20 sm:py-24">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div>
              <div className="in-label">live, from the running model</div>
              <h2 className="in-display text-3xl sm:text-4xl mt-4 max-w-2xl">
                Watch the interval widen as the molecule gets stranger.
              </h2>
              <p
                className="text-sm mt-3 max-w-xl"
                style={{ color: 'var(--pp-muted)' }}
              >
                These are real requests to <code className="in-num">/api/predict</code>,
                made by your browser just now. Nothing on this page is stored
                or typed in.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              {SHOWCASE.map((c) => {
                const on = picked.name === c.name;
                return (
                  <button
                    key={c.name}
                    onClick={() => setPicked(c)}
                    className="px-3.5 py-2 text-[12px] rounded-full transition-colors cursor-pointer"
                    style={{
                      background: on ? 'var(--pp-ink)' : 'var(--pp-raised)',
                      color: on ? 'var(--pp-paper)' : 'var(--pp-ink)',
                      border: `1px solid ${on ? 'var(--pp-ink)' : 'var(--pp-rule)'}`,
                    }}
                  >
                    {c.name}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-10 in-panel p-7 sm:p-9">
            {demoError && (
              <div className="text-sm" style={{ color: 'var(--pp-alarm)' }}>
                {demoError}
              </div>
            )}

            {!demoError && (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                <div className="lg:col-span-5">
                  <h3 className="in-display text-2xl">{picked.name}</h3>
                  <div
                    className="in-num text-[11px] mt-2 break-all"
                    style={{ color: 'var(--pp-muted)' }}
                  >
                    {picked.smiles}
                  </div>

                  {demo?.descriptors && (
                    <dl className="mt-6 grid grid-cols-2 gap-y-4">
                      {[
                        ['formula', demo.descriptors.formula],
                        ['MW', demo.descriptors.molecular_weight],
                        ['logP', demo.descriptors.logp],
                        ['QED', demo.descriptors.qed],
                      ].map(([k, v]) => (
                        <div key={String(k)}>
                          <dt className="in-label">{k}</dt>
                          <dd className="in-num text-base mt-1">{String(v)}</dd>
                        </div>
                      ))}
                    </dl>
                  )}

                  <button
                    onClick={() => onLaunchApp('prediction')}
                    className="in-btn mt-7"
                  >
                    Take it to the workbench
                  </button>
                </div>

                <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-5">
                  <div className="pp-well p-5">
                    <div className="in-label">predicted solubility</div>
                    <div className="in-num text-4xl font-light mt-3">
                      {iv90 && typeof sol?.value === 'number' ? (
                        <Uncertain
                          sigma={iv90.margin_error}
                          outOfDomain={domain?.status === 'OUT_OF_DOMAIN'}
                        >
                          {sol.value.toFixed(3)}
                        </Uncertain>
                      ) : demoLoading ? (
                        '…'
                      ) : (
                        '—'
                      )}
                    </div>
                    <div
                      className="text-[11px] mt-2"
                      style={{ color: 'var(--pp-faint)' }}
                    >
                      log(mol/L) · {sol?.model ?? 'model unavailable'}
                    </div>
                  </div>

                  <div className="pp-well p-5">
                    <div className="in-label">90% conformal interval</div>
                    <div className="in-num text-2xl font-light mt-3">
                      {iv90
                        ? `[${iv90.lower_bound.toFixed(2)}, ${iv90.upper_bound.toFixed(2)}]`
                        : '—'}
                    </div>
                    <div
                      className="text-[11px] mt-2"
                      style={{ color: 'var(--pp-faint)' }}
                    >
                      {iv90
                        ? `width ${iv90.interval_width.toFixed(2)} · calibrated on ${sol?.conformal?.calibration_set_size} held-out compounds`
                        : 'awaiting backend'}
                    </div>
                  </div>

                  <div className="pp-well p-5">
                    <div className="in-label">applicability domain</div>
                    <div
                      className="in-display text-lg mt-3 flex items-center gap-2"
                      style={{
                        color:
                          DOMAIN_COLOR[domain?.status] ?? 'var(--pp-muted)',
                      }}
                    >
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{
                          background:
                            DOMAIN_COLOR[domain?.status] ?? 'var(--pp-faint)',
                        }}
                      />
                      {domain?.status ?? '—'}
                    </div>
                    <div
                      className="text-[11px] mt-2"
                      style={{ color: 'var(--pp-faint)' }}
                    >
                      Tanimoto similarity to the training set combined with PCA
                      leverage
                    </div>
                  </div>

                  <div className="pp-well p-5">
                    <div className="in-label">ensemble spread</div>
                    <div className="in-num text-2xl font-light mt-3">
                      {typeof sol?.ensemble_spread === 'number'
                        ? sol.ensemble_spread.toFixed(3)
                        : '—'}
                    </div>
                    <div
                      className="text-[11px] mt-2"
                      style={{ color: 'var(--pp-faint)' }}
                    >
                      disagreement between the trees, before calibration
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <p
            className="in-label mt-5"
            style={{ color: 'var(--pp-faint)', textTransform: 'none', letterSpacing: '0.02em' }}
          >
            The number above blurs in proportion to its own interval. That is
            the one visual rule the whole interface is built on.
          </p>
        </div>
      </section>

      {/* ----------------------------------------------------- modules */}
      <section
        id="modules"
        className="max-w-[1400px] w-full mx-auto px-5 sm:px-8 py-20 sm:py-28"
      >
        <Rise>
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div>
              <div className="in-label">what is in it</div>
              <h2 className="in-display text-3xl sm:text-5xl mt-4 max-w-2xl">
                Six modules, one pipeline.
              </h2>
            </div>
            <button onClick={() => onLaunchApp('analyzer')} className="in-btn self-start">
              Open all six
            </button>
          </div>
        </Rise>

        <div className="mt-14">
          {MODULES.map((m) => (
            <button
              key={m.num}
              onClick={() => onLaunchApp(m.id)}
              className="w-full text-left group cursor-pointer"
              style={{ borderTop: '1px solid var(--pp-rule)' }}
            >
              <div className="py-7 grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-8 items-baseline transition-opacity group-hover:opacity-100 md:opacity-90">
                <div className="md:col-span-1 in-num text-sm" style={{ color: 'var(--pp-faint)' }}>
                  {m.num}
                </div>
                <div className="md:col-span-3">
                  <h3 className="in-display text-2xl group-hover:underline underline-offset-4 decoration-1">
                    {m.title}
                  </h3>
                </div>
                <div className="md:col-span-4">
                  <p className="text-sm leading-relaxed">{m.what}</p>
                </div>
                <div className="md:col-span-4">
                  <p
                    className="text-[12.5px] leading-relaxed"
                    style={{ color: 'var(--pp-muted)' }}
                  >
                    {m.how}
                  </p>
                </div>
              </div>
            </button>
          ))}
          <div style={{ borderTop: '1px solid var(--pp-rule)' }} />
        </div>
      </section>

      {/* -------------------------------------------------- validation */}
      <section
        id="validation"
        style={{
          background: 'var(--pp-sunken)',
          borderTop: '1px solid var(--pp-rule)',
        }}
      >
        <div className="max-w-[1400px] w-full mx-auto px-5 sm:px-8 py-20 sm:py-24">
          <Rise>
            <div className="in-label">validation</div>
            <h2 className="in-display text-3xl sm:text-4xl mt-4 max-w-2xl">
              Measured against data the model never saw.
            </h2>
            <p
              className="text-sm mt-4 max-w-2xl"
              style={{ color: 'var(--pp-muted)' }}
            >
              The solubility model is trained on the Delaney ESOL set of
              measured aqueous solubilities. The figures here come from the
              held-out test partition and from conformal calibration on a
              separate slice, fetched from{' '}
              <code className="in-num">/api/benchmark</code> rather than written
              into this page.
            </p>
          </Rise>

          <div className="mt-12 grid grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                v: fmt(metrics?.r2_score),
                t: 'test R²',
                n: 'coefficient of determination on the held-out split',
              },
              {
                v: fmt(metrics?.rmse),
                t: 'RMSE',
                n: 'log(mol/L), same split',
              },
              {
                v: fmt(metrics?.mae),
                t: 'MAE',
                n: 'mean absolute error, log(mol/L)',
              },
              {
                v:
                  typeof coverage90 === 'number'
                    ? `${coverage90.toFixed(1)}%`
                    : '—',
                t: 'coverage at 90%',
                n: 'how often the interval actually contained the measurement',
              },
            ].map((s) => (
              <div key={s.t} className="in-panel p-6">
                <div className="in-num text-3xl sm:text-4xl font-light">
                  {s.v}
                </div>
                <div className="in-label mt-3">{s.t}</div>
                <div
                  className="text-[11px] mt-2 leading-relaxed"
                  style={{ color: 'var(--pp-faint)' }}
                >
                  {s.n}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 in-panel p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-5">
            <p className="text-sm" style={{ color: 'var(--pp-muted)' }}>
              The benchmark module shows the parity plot, the residual
              distribution and the coverage table behind these four numbers.
            </p>
            <button
              onClick={() => onLaunchApp('benchmark')}
              className="in-btn shrink-0"
            >
              Open benchmark
            </button>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------ footer */}
      <footer
        className="mt-auto"
        style={{ borderTop: '1px solid var(--pp-rule)', background: 'var(--pp-paper)' }}
      >
        <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">
          <div>
            <div className="in-display text-sm">MoleculeMind</div>
            <div className="in-label mt-1.5">
              RDKit · scikit-learn · PyTorch · BoTorch · FastAPI
            </div>
          </div>
          <div className="flex items-center gap-6">
            <button onClick={onOpenGlossary} className="in-label hover:opacity-60 transition-opacity cursor-pointer">
              methodology
            </button>
            <button
              onClick={() => onLaunchApp('analyzer')}
              className="in-label hover:opacity-60 transition-opacity cursor-pointer"
              style={{ color: 'var(--pp-accent)' }}
            >
              open workbench →
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
};
