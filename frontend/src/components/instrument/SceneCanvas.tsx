/**
 * MoleculeFigure — the 3D molecule, as a figure in a document.
 *
 * This replaces an earlier design where a full-screen WebGL canvas sat behind
 * the whole application. That version measured **4 FPS**. The cause was not the
 * 3D itself but the combination: a continuously animating full-viewport canvas
 * underneath fourteen panels that each used `backdrop-filter: blur`, so the
 * browser recomputed fourteen blurred regions every single frame.
 *
 * The canvas is now contained, modest, and only mounted where a molecule is
 * actually the subject. Surfaces above it are solid. Specific measures:
 *
 *   - device pixel ratio capped at 1.5 (was up to 1.75 on a 2x display, which
 *     meant a 3.2-megapixel backing store)
 *   - the ambient particle field is gone entirely
 *   - rendering pauses when the figure scrolls out of view or the tab is hidden
 *   - no post-processing, no shadows, two lights
 *
 * The state provider is kept so modules can still report a molecule and its
 * uncertainty, which drives both the figure and the typographic blur.
 */

import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export type SceneMode =
  | 'idle'
  | 'analyzer'
  | 'prediction'
  | 'pareto'
  | 'discovery'
  | 'active_learning'
  | 'benchmark';

export interface Atom {
  element: string;
  x: number;
  y: number;
  z: number;
}

interface SceneApi {
  mode: SceneMode;
  uncertainty: number;
  outOfDomain: boolean;
  atoms: Atom[];
  bonds: [number, number][];
  setMode: (m: SceneMode) => void;
  setUncertainty: (u: number) => void;
  setOutOfDomain: (v: boolean) => void;
  setMolecule: (atoms: Atom[], bonds: [number, number][]) => void;
}

const SceneContext = createContext<SceneApi | null>(null);

export const useScene = (): SceneApi => {
  const ctx = useContext(SceneContext);
  if (!ctx) throw new Error('useScene must be used inside <SceneProvider>');
  return ctx;
};

/**
 * Element colours chosen for a LIGHT background. The usual CPK palette assumes
 * dark: white hydrogens and pale greys vanish on paper, so carbons are graphite
 * and hydrogens a mid grey.
 */
const ELEMENT_COLOR: Record<string, string> = {
  C: '#3A3A42',
  H: '#B8B8BE',
  O: '#C0392B',
  N: '#2B5FD9',
  S: '#C8901B',
  P: '#C06A1B',
  F: '#3E8E5A',
  Cl: '#4A8C4A',
  Br: '#8C5A2B',
  I: '#6B3FA0',
};
const ELEMENT_RADIUS: Record<string, number> = {
  H: 0.2, C: 0.32, N: 0.3, O: 0.29, S: 0.38, P: 0.38,
  F: 0.24, Cl: 0.36, Br: 0.4, I: 0.44,
};

const colorFor = (el: string) => ELEMENT_COLOR[el] ?? '#3A3A42';
const radiusFor = (el: string) => ELEMENT_RADIUS[el] ?? 0.31;

const Molecule: React.FC<{
  atoms: Atom[];
  bonds: [number, number][];
  spin: boolean;
}> = ({ atoms, bonds, spin }) => {
  const group = useRef<THREE.Group>(null);

  const { positions, scale } = useMemo(() => {
    if (!atoms.length) return { positions: [] as THREE.Vector3[], scale: 1 };
    const pts = atoms.map((a) => new THREE.Vector3(a.x, a.y, a.z));
    const centre = pts
      .reduce((acc, p) => acc.add(p), new THREE.Vector3())
      .divideScalar(pts.length);
    pts.forEach((p) => p.sub(centre));
    const extent = Math.max(...pts.map((p) => p.length()), 1);
    return { positions: pts, scale: 3.1 / extent };
  }, [atoms]);

  useFrame((_, delta) => {
    if (group.current && spin) group.current.rotation.y += delta * 0.22;
  });

  if (!atoms.length) return null;

  return (
    <group ref={group} scale={scale}>
      {positions.map((p, i) => (
        <mesh key={`a${i}`} position={p}>
          {/* Segment counts were cut to 16 when the canvas was full-screen.
              The figure is a fraction of that area now, so the triangles are
              cheap again and the faceting they caused was the main reason the
              spheres read as low quality. */}
          <sphereGeometry args={[radiusFor(atoms[i].element), 40, 28]} />
          <meshStandardMaterial
            color={colorFor(atoms[i].element)}
            roughness={0.38}
            metalness={0.04}
          />
        </mesh>
      ))}
      {bonds.map(([a, b], i) => {
        const start = positions[a];
        const end = positions[b];
        if (!start || !end) return null;
        const mid = start.clone().add(end).multiplyScalar(0.5);
        const dir = end.clone().sub(start);
        const quat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir.clone().normalize()
        );
        return (
          <mesh key={`b${i}`} position={mid} quaternion={quat}>
            <cylinderGeometry args={[0.075, 0.075, dir.length(), 20]} />
            <meshStandardMaterial color="#A6A6AE" roughness={0.45} />
          </mesh>
        );
      })}
    </group>
  );
};

/**
 * The molecule as a figure. Place it in a layout like an illustration.
 *
 * It renders nothing at all until it is on screen, and stops rendering when it
 * leaves — the single most effective thing you can do for a WebGL canvas that
 * shares a page with scrolling content.
 */
export const MoleculeFigure: React.FC<{
  className?: string;
  height?: number;
  caption?: string;
}> = ({ className, height = 280, caption }) => {
  const { atoms, bonds, uncertainty, outOfDomain } = useScene();
  const host = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = host.current;
    if (!el || typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return;
    }
    const io = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: '120px' }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Pause while the browser tab is in the background.
  const [tabActive, setTabActive] = useState(true);
  useEffect(() => {
    const onVis = () => setTabActive(!document.hidden);
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  const running = visible && tabActive;

  return (
    <figure ref={host} className={className} style={{ margin: 0 }}>
      <div
        className="pp-well overflow-hidden"
        style={{
          height,
          // The signature rule, applied to the figure.
          //
          // This used to blur the render in proportion to uncertainty, which
          // was a mistake: a blurred 3D render is indistinguishable from a
          // badly rendered one, so the rule read as poor image quality rather
          // than as a statement about confidence. The numbers still blur —
          // there the intent is unmistakable, and they carry a tooltip. Here
          // the figure fades and desaturates instead, which reads as
          // "provisional" while staying sharp.
          opacity: outOfDomain ? 0.55 : 1 - uncertainty * 0.22,
          filter: outOfDomain
            ? 'saturate(0.25) contrast(0.9)'
            : `saturate(${(1 - uncertainty * 0.45).toFixed(2)})`,
          transition:
            'opacity 0.6s cubic-bezier(0.22, 1, 0.36, 1), filter 0.6s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        {atoms.length > 0 && (
          <Canvas
            camera={{ position: [0, 0, 9], fov: 34 }}
            // 1.5 was a full-screen-canvas compromise. At figure size the
            // backing store is small enough to render at the display's own
            // pixel ratio, which is what stops the edges looking soft on a
            // Retina screen.
            dpr={[1, 2]}
            // `demand` would need manual invalidation for the spin; `always`
            // is fine because the canvas unmounts its loop when not running.
            frameloop={running ? 'always' : 'never'}
            gl={{ antialias: true, alpha: true, powerPreference: 'default' }}
            style={{ pointerEvents: 'none' }}
          >
            <ambientLight intensity={0.7} />
            <directionalLight position={[4, 6, 8]} intensity={1.35} />
            {/* A cool fill from the opposite side stops the unlit hemispheres
                going flat grey against a light page. */}
            <directionalLight position={[-6, -3, -4]} intensity={0.45} color="#C9D4FF" />
            <Molecule atoms={atoms} bonds={bonds} spin={running} />
          </Canvas>
        )}
      </div>
      {caption && (
        <figcaption className="in-label mt-2.5">{caption}</figcaption>
      )}
    </figure>
  );
};

export const SceneProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [mode, setMode] = useState<SceneMode>('idle');
  const [uncertainty, setUncertainty] = useState(0);
  const [outOfDomain, setOutOfDomain] = useState(false);
  const [atoms, setAtoms] = useState<Atom[]>([]);
  const [bonds, setBonds] = useState<[number, number][]>([]);

  const api = useMemo<SceneApi>(
    () => ({
      mode,
      uncertainty,
      outOfDomain,
      atoms,
      bonds,
      setMode,
      setUncertainty,
      setOutOfDomain,
      setMolecule: (a, b) => {
        setAtoms(a);
        setBonds(b);
      },
    }),
    [mode, uncertainty, outOfDomain, atoms, bonds]
  );

  // No fixed canvas, no scrim, no grain. The provider is state only.
  return <SceneContext.Provider value={api}>{children}</SceneContext.Provider>;
};

/**
 * Parse RDKit's MOL V2000 block into atoms and bonds.
 *
 * Line 4 is the counts line, then one line per atom (x, y, z, element), then
 * one line per bond using 1-based atom indices.
 */
export function parseMolBlock(mol: string): { atoms: Atom[]; bonds: [number, number][] } {
  const empty = { atoms: [] as Atom[], bonds: [] as [number, number][] };
  if (!mol) return empty;

  const lines = mol.split('\n');
  if (lines.length < 5) return empty;

  const counts = lines[3];
  const nAtoms = parseInt(counts.slice(0, 3).trim(), 10);
  const nBonds = parseInt(counts.slice(3, 6).trim(), 10);
  if (!Number.isFinite(nAtoms) || !Number.isFinite(nBonds)) return empty;

  const atoms: Atom[] = [];
  for (let i = 0; i < nAtoms; i++) {
    const line = lines[4 + i];
    if (!line) break;
    const x = parseFloat(line.slice(0, 10));
    const y = parseFloat(line.slice(10, 20));
    const z = parseFloat(line.slice(20, 30));
    const element = line.slice(31, 34).trim();
    if ([x, y, z].every(Number.isFinite)) atoms.push({ element, x, y, z });
  }

  const bonds: [number, number][] = [];
  for (let i = 0; i < nBonds; i++) {
    const line = lines[4 + nAtoms + i];
    if (!line) break;
    const a = parseInt(line.slice(0, 3).trim(), 10) - 1;
    const b = parseInt(line.slice(3, 6).trim(), 10) - 1;
    if (a >= 0 && b >= 0 && a < atoms.length && b < atoms.length) bonds.push([a, b]);
  }

  return { atoms, bonds };
}
