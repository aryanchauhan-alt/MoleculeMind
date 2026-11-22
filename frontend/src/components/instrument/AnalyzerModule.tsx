/**
 * AnalyzerModule — module 01, rebuilt on the Instrument system.
 *
 * The important difference from the old tab: the molecule is not rendered in a
 * little box inside the panel. It is pushed to the application's shared WebGL
 * canvas, so it floats behind the whole interface and persists when you move to
 * another module. The panels are read as instruments laid over it.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowRight, Loader2, TriangleAlert } from 'lucide-react';
import {
  CountUp,
  DUR,
  EASE,
  Magnetic,
  RevealText,
  Rise,
  Stagger,
  StaggerItem,
} from './primitives';
import { parseMolBlock, useScene } from './SceneCanvas';

const PRESETS = [
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Ibuprofen', smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' },
  { name: 'Warfarin', smiles: 'CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O' },
  // Kept deliberately: this is the molecule that defeats the similarity test.
  { name: 'C36 chain', smiles: 'C'.repeat(36) },
];

interface Props {
  onSendToPrediction?: (smiles: string) => void;
}

export const AnalyzerModule: React.FC<Props> = ({ onSendToPrediction }) => {
  const scene = useScene();
  const [smiles, setSmiles] = useState(PRESETS[0].smiles);
  const [result, setResult] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyse = useCallback(
    async (target?: string) => {
      const query = (target ?? smiles).trim();
      if (!query) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ smiles: query }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Could not analyse that structure.');
        setResult(json);

        // Hand the conformer to the shared canvas.
        if (json.sdf_3d) {
          const { atoms, bonds } = parseMolBlock(json.sdf_3d);
          scene.setMolecule(atoms, bonds);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not analyse that structure.');
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [smiles, scene]
  );

  // Set the scene mode on mount, and analyse the default molecule once.
  useEffect(() => {
    scene.setMode('analyzer');
    scene.setUncertainty(0);
    scene.setOutOfDomain(false);
    analyse(PRESETS[0].smiles);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const d = result?.descriptors;

  return (
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-14">
      {/* --- Title ------------------------------------------------------ */}
      <div className="mb-12">
        <div className="in-label mb-4">module 01 — molecular analyzer</div>
        <h1 className="in-display-xl text-[11vw] sm:text-[5.5vw] leading-[0.9]">
          <RevealText text="Read the structure" />
        </h1>
        <div className="in-rule mt-8 max-w-md" />
        <Rise delay={0.4}>
          <p
            className="mt-5 text-sm max-w-xl leading-relaxed"
            style={{ color: 'var(--in-dim)' }}
          >
            A molecule enters as text. RDKit parses it, standardises the spelling, and
            computes its physical properties. The structure you see behind this panel is a
            computed low-energy conformer — not an experimentally measured shape.
          </p>
        </Rise>
      </div>

      {/* --- Input ------------------------------------------------------ */}
      <Rise delay={0.2} className="mb-10">
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && analyse()}
            spellCheck={false}
            placeholder="Enter a SMILES string"
            className="in-input flex-1 px-5 py-4 text-sm"
            style={{ color: 'var(--in-text)' }}
          />
          <Magnetic strength={4}>
            <button
              onClick={() => analyse()}
              disabled={loading}
              className="in-btn px-7 py-4 w-full sm:w-auto cursor-pointer disabled:opacity-40 flex items-center justify-center gap-2"
              style={{ background: 'var(--in-signal)', color: '#04161A', fontWeight: 600 }}
            >
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
              {loading ? 'reading' : 'analyse'}
            </button>
          </Magnetic>
        </div>

        <div className="flex flex-wrap gap-2 mt-4">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => {
                setSmiles(p.smiles);
                analyse(p.smiles);
              }}
              className="in-btn px-3.5 py-1.5 cursor-pointer"
              style={{
                border: '1px solid var(--in-line)',
                color: 'var(--in-dim)',
                background: 'rgba(0,0,0,0.3)',
              }}
            >
              {p.name}
            </button>
          ))}
        </div>
      </Rise>

      {/* --- Error ------------------------------------------------------ */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            className="in-panel p-5 mb-8 flex gap-3 items-start"
            style={{ borderColor: 'rgba(240,113,91,0.25)' }}
          >
            <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" style={{ color: 'var(--in-alarm)' }} />
            <div>
              <div className="in-display text-sm" style={{ color: 'var(--in-alarm)' }}>
                Invalid structure
              </div>
              <p className="text-xs mt-1" style={{ color: 'var(--in-dim)' }}>
                {error}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* --- Readouts --------------------------------------------------- */}
      {d && (
        <>
          <Stagger className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-3" delay={0.1}>
            {[
              { label: 'formula', value: d.formula, mono: true },
              { label: 'molecular weight', value: d.molecular_weight, unit: 'g/mol', count: 2 },
              { label: 'logP', value: d.logp, count: 2 },
              { label: 'TPSA', value: d.tpsa, unit: 'Å²', count: 2 },
            ].map((item) => (
              <StaggerItem key={item.label}>
                <div className="in-panel in-panel-hover p-5 h-full">
                  <div className="in-label">{item.label}</div>
                  <div
                    className="in-num text-2xl font-light mt-2"
                    style={{ color: 'var(--in-signal)' }}
                  >
                    {item.count !== undefined ? (
                      <CountUp value={Number(item.value)} decimals={item.count} />
                    ) : (
                      item.value
                    )}
                    {item.unit && (
                      <span className="text-[10px] ml-1.5" style={{ color: 'var(--in-faint)' }}>
                        {item.unit}
                      </span>
                    )}
                  </div>
                </div>
              </StaggerItem>
            ))}
          </Stagger>

          <Stagger className="grid grid-cols-3 lg:grid-cols-6 gap-3 mb-10" delay={0.2}>
            {[
              { label: 'QED', value: d.qed, count: 3 },
              { label: 'H donors', value: d.hbd, count: 0 },
              { label: 'H acceptors', value: d.hba, count: 0 },
              { label: 'rotatable', value: d.rotatable_bonds, count: 0 },
              { label: 'rings', value: d.ring_count, count: 0 },
              { label: 'heavy atoms', value: d.heavy_atom_count, count: 0 },
            ].map((item) => (
              <StaggerItem key={item.label}>
                <div className="in-panel in-panel-hover p-4 h-full">
                  <div className="in-label">{item.label}</div>
                  <div className="in-num text-lg font-light mt-1.5">
                    <CountUp value={Number(item.value)} decimals={item.count} />
                  </div>
                </div>
              </StaggerItem>
            ))}
          </Stagger>

          {/* --- Rules ---------------------------------------------------- */}
          <Rise delay={0.3}>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
              {[
                {
                  title: "Lipinski's Rule of Five",
                  passed: d.lipinski.passed,
                  detail:
                    d.lipinski.violations.length > 0
                      ? d.lipinski.violations.join(' · ')
                      : 'No violations. Within the usual range for an oral drug.',
                  note: `${d.lipinski.violations_count} of 4 criteria exceeded`,
                },
                {
                  title: 'Veber Rule',
                  passed: d.veber.passed,
                  detail: `Rotatable bonds ${d.rotatable_bonds} ≤ 10 · TPSA ${d.tpsa} ≤ 140`,
                  note: 'Oral bioavailability heuristic',
                },
              ].map((rule) => (
                <div key={rule.title} className="in-panel p-6">
                  <div className="flex items-start justify-between gap-3">
                    <div className="in-display text-sm">{rule.title}</div>
                    <span
                      className="in-btn px-2.5 py-1 shrink-0"
                      style={{
                        background: rule.passed
                          ? 'rgba(76,224,224,0.1)'
                          : 'rgba(240,179,91,0.1)',
                        color: rule.passed ? 'var(--in-signal)' : 'var(--in-caution)',
                        border: `1px solid ${
                          rule.passed ? 'rgba(76,224,224,0.22)' : 'rgba(240,179,91,0.22)'
                        }`,
                      }}
                    >
                      {rule.passed ? 'pass' : 'flag'}
                    </span>
                  </div>
                  <p className="text-xs mt-3 leading-relaxed" style={{ color: 'var(--in-dim)' }}>
                    {rule.detail}
                  </p>
                  <div className="in-label mt-4">{rule.note}</div>
                </div>
              ))}

              {/* 2D structure sits beside the rules; the 3D is on the shared canvas. */}
              <div className="in-panel p-6 flex flex-col">
                <div className="in-label mb-3">2D structure</div>
                <div
                  className="flex-1 flex items-center justify-center rounded-xl overflow-hidden"
                  style={{ background: 'rgba(0,0,0,0.25)', minHeight: 150 }}
                  dangerouslySetInnerHTML={{ __html: result.svg_2d }}
                />
                <div className="in-num text-[10px] mt-3 truncate" style={{ color: 'var(--in-faint)' }}>
                  {result.smiles}
                </div>
              </div>
            </div>
          </Rise>

          {/* --- Conformer note + handoff ---------------------------------- */}
          <Rise delay={0.4}>
            <div className="in-panel p-6 mt-3 flex flex-col sm:flex-row sm:items-center justify-between gap-5">
              <div className="flex-1">
                <div className="in-label">3D conformer</div>
                <p className="text-xs mt-2 leading-relaxed max-w-2xl" style={{ color: 'var(--in-dim)' }}>
                  {result.sdf_error
                    ? `No conformer could be generated: ${result.sdf_error}`
                    : result.sdf_note}
                </p>
              </div>
              {onSendToPrediction && !result.sdf_error && (
                <Magnetic strength={4}>
                  <button
                    onClick={() => onSendToPrediction(result.smiles)}
                    className="in-btn px-6 py-3 cursor-pointer flex items-center gap-2 whitespace-nowrap"
                    style={{
                      border: '1px solid var(--in-line-strong)',
                      color: 'var(--in-text)',
                    }}
                  >
                    predict properties
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </Magnetic>
              )}
            </div>
          </Rise>
        </>
      )}
    </div>
  );
};
