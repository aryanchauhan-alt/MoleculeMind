/**
 * SceneCanvas — one WebGL canvas for the entire application.
 *
 * This is the architectural choice that creates the "continuous" feel. The
 * canvas mounts once and never unmounts. Switching modules does not reload
 * anything: the camera flies, the molecule morphs, and the HTML panels
 * cross-fade on top. Nothing ever pops.
 *
 * The signature rule of the design system is enforced here too: UNCERTAINTY IS
 * OPTICAL. A confident prediction renders as a crisp, tightly-glowing molecule.
 * An uncertain one renders soft, desaturated and surrounded by a thick
 * probability shell whose radius is the real conformal interval width.
 */

import React, { createContext, useContext, useMemo, useRef, useState } from 'react';
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

export interface SceneState {
  mode: SceneMode;
  /** 0 = fully certain, 1 = no idea. Drives blur, glow spread and shell size. */
  uncertainty: number;
  /** Set when the applicability domain rejects the molecule. */
  outOfDomain: boolean;
  /** Atom positions in Angstroms, parsed from the backend's 3D conformer. */
  atoms: Atom[];
  bonds: [number, number][];
}

export interface Atom {
  element: string;
  x: number;
  y: number;
  z: number;
}

interface SceneApi extends SceneState {
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

/** CPK-ish colours, warmed slightly to sit inside the palette. */
const ELEMENT_COLOR: Record<string, string> = {
  C: '#8A93A3',
  H: '#E8ECF2',
  O: '#F0715B',
  N: '#4CE0E0',
  S: '#F0B35B',
  P: '#F09A5B',
  F: '#7BE07B',
  Cl: '#7BE09A',
  Br: '#C98A5B',
  I: '#A87BE0',
};
const ELEMENT_RADIUS: Record<string, number> = {
  H: 0.22, C: 0.34, N: 0.32, O: 0.31, S: 0.4, P: 0.4,
  F: 0.26, Cl: 0.38, Br: 0.42, I: 0.46,
};

const colorFor = (el: string) => ELEMENT_COLOR[el] ?? '#8A93A3';
const radiusFor = (el: string) => ELEMENT_RADIUS[el] ?? 0.33;

/**
 * Camera framing per module. The camera never cuts — it eases between these,
 * which is most of what makes module changes feel cinematic rather than like
 * page loads.
 */
const CAMERA_TARGETS: Record<SceneMode, { pos: [number, number, number]; fov: number }> = {
  idle:            { pos: [0, 0, 16], fov: 38 },
  analyzer:        { pos: [0, 0, 11], fov: 34 },
  prediction:      { pos: [3.5, 1.2, 12], fov: 36 },
  pareto:          { pos: [0, 2.5, 22], fov: 44 },
  discovery:       { pos: [-3, 1.5, 19], fov: 42 },
  active_learning: { pos: [0, -1.5, 20], fov: 46 },
  benchmark:       { pos: [8, 3, 26], fov: 50 },
};

/**
 * Where the molecule sits per module, in world units. It lives in the right
 * third on text-heavy modules so the reading column stays clear, and recentres
 * where the 3D itself is the subject.
 */
const MOLECULE_OFFSET: Record<SceneMode, [number, number, number]> = {
  idle:            [0, 0, 0],
  analyzer:        [3.4, 0.2, 0],
  prediction:      [3.2, 0.1, 0],
  pareto:          [5.0, 1.0, -3],
  discovery:       [5.2, 0.8, -3],
  active_learning: [4.6, 0.6, -2],
  benchmark:       [6.0, 1.4, -5],
};

/** Eases the camera toward the active module's framing every frame. */
const CameraRig: React.FC<{ mode: SceneMode }> = ({ mode }) => {
  const target = CAMERA_TARGETS[mode] ?? CAMERA_TARGETS.idle;
  const pointer = useRef({ x: 0, y: 0 });

  useFrame((state, delta) => {
    const cam = state.camera as THREE.PerspectiveCamera;
    // Critically damped approach: fast at first, settles without overshoot.
    const k = 1 - Math.exp(-delta * 2.2);

    // A little parallax from the pointer. Subtle — 0.8 units at the extremes.
    pointer.current.x += (state.pointer.x * 0.8 - pointer.current.x) * k;
    pointer.current.y += (state.pointer.y * 0.5 - pointer.current.y) * k;

    cam.position.x += (target.pos[0] + pointer.current.x - cam.position.x) * k;
    cam.position.y += (target.pos[1] + pointer.current.y - cam.position.y) * k;
    cam.position.z += (target.pos[2] - cam.position.z) * k;

    if (Math.abs(cam.fov - target.fov) > 0.01) {
      cam.fov += (target.fov - cam.fov) * k;
      cam.updateProjectionMatrix();
    }
    cam.lookAt(0, 0, 0);
  });

  return null;
};

/**
 * The molecule itself: instanced spheres for atoms, cylinders for bonds.
 * Instancing matters — a 40-atom molecule with individual meshes costs 40 draw
 * calls per frame; instanced it costs one.
 */
const Molecule: React.FC<{
  atoms: Atom[];
  bonds: [number, number][];
  uncertainty: number;
  offset: [number, number, number];
}> = ({ atoms, bonds, uncertainty, offset }) => {
  const group = useRef<THREE.Group>(null);
  const pos = useRef(new THREE.Vector3(...offset));

  // Centre and scale the molecule so any size sits nicely in frame.
  const { positions, scale } = useMemo(() => {
    if (!atoms.length) return { positions: [] as THREE.Vector3[], scale: 1 };
    const pts = atoms.map((a) => new THREE.Vector3(a.x, a.y, a.z));
    const centre = pts
      .reduce((acc, p) => acc.add(p), new THREE.Vector3())
      .divideScalar(pts.length);
    pts.forEach((p) => p.sub(centre));
    const extent = Math.max(...pts.map((p) => p.length()), 1);
    // Target a consistent on-screen size regardless of molecule size. Kept
    // deliberately modest: the molecule is atmosphere, the panels are the
    // subject, and a molecule large enough to read through is a molecule that
    // makes the text unreadable.
    return { positions: pts, scale: 2.1 / extent };
  }, [atoms]);

  useFrame((_, delta) => {
    if (!group.current) return;
    // Slow, continuous rotation. Uncertain molecules wobble very slightly —
    // an unsettled object reads as an unsettled answer.
    group.current.rotation.y += delta * 0.16;
    group.current.rotation.x =
      Math.sin(performance.now() * 0.0004) * (0.06 + uncertainty * 0.12);

    // Ease toward the module's framing offset rather than cutting.
    const k = 1 - Math.exp(-delta * 1.8);
    pos.current.x += (offset[0] - pos.current.x) * k;
    pos.current.y += (offset[1] - pos.current.y) * k;
    pos.current.z += (offset[2] - pos.current.z) * k;
    group.current.position.copy(pos.current);
  });

  if (!atoms.length) return null;

  return (
    <group ref={group} scale={scale} position={offset}>
      {positions.map((p, i) => (
        <mesh key={`a${i}`} position={p}>
          <sphereGeometry args={[radiusFor(atoms[i].element), 24, 24]} />
          <meshStandardMaterial
            color={colorFor(atoms[i].element)}
            roughness={0.35}
            metalness={0.1}
            emissive={colorFor(atoms[i].element)}
            emissiveIntensity={0.16}
          />
        </mesh>
      ))}

      {bonds.map(([a, b], i) => {
        const start = positions[a];
        const end = positions[b];
        if (!start || !end) return null;
        const mid = start.clone().add(end).multiplyScalar(0.5);
        const dir = end.clone().sub(start);
        const len = dir.length();
        // Align a Y-up cylinder along the bond vector.
        const quat = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir.clone().normalize()
        );
        return (
          <mesh key={`b${i}`} position={mid} quaternion={quat}>
            <cylinderGeometry args={[0.075, 0.075, len, 12]} />
            <meshStandardMaterial color="#5A6372" roughness={0.5} metalness={0.2} />
          </mesh>
        );
      })}
    </group>
  );
};

/**
 * The probability shell — the clearest expression of the project's thesis.
 * Its radius grows with the calibrated uncertainty, so a confident prediction
 * wears a tight skin and an unsure one sits inside a thick fog.
 */
const UncertaintyShell: React.FC<{
  uncertainty: number;
  outOfDomain: boolean;
  offset: [number, number, number];
}> = ({ uncertainty, outOfDomain, offset }) => {
  const mesh = useRef<THREE.Mesh>(null);
  const current = useRef(0);

  useFrame((_, delta) => {
    if (!mesh.current) return;
    const k = 1 - Math.exp(-delta * 1.6);
    const target = 2.4 + uncertainty * 2.6;
    current.current += (target - current.current) * k;
    mesh.current.scale.setScalar(current.current);
    mesh.current.rotation.y -= delta * 0.05;
  });

  return (
    <mesh ref={mesh} position={offset}>
      <icosahedronGeometry args={[1, 3]} />
      <meshBasicMaterial
        color={outOfDomain ? '#F0715B' : '#4CE0E0'}
        wireframe
        transparent
        opacity={0.04 + uncertainty * 0.12}
      />
    </mesh>
  );
};

/** Ambient particle field. Gives the empty space depth and parallax. */
const Starfield: React.FC<{ count?: number }> = ({ count = 420 }) => {
  const points = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Shell distribution so nothing sits in front of the molecule.
      const r = 20 + Math.random() * 30;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [count]);

  useFrame((_, delta) => {
    if (points.current) points.current.rotation.y += delta * 0.012;
  });

  return (
    <points ref={points} geometry={geometry}>
      <pointsMaterial size={0.08} color="#4CE0E0" transparent opacity={0.3} sizeAttenuation />
    </points>
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

  return (
    <SceneContext.Provider value={api}>
      {/* The canvas is fixed behind everything and never unmounts. */}
      <div
        className="fixed inset-0 z-0 pointer-events-none"
        style={{
          // The signature rule, applied to the whole 3D layer.
          filter: outOfDomain
            ? 'blur(2px) saturate(0.4)'
            : `blur(${uncertainty * 1.6}px) saturate(${1 - uncertainty * 0.3})`,
          transition: 'filter 1.2s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        <Canvas
          camera={{ position: [0, 0, 16], fov: 38 }}
          dpr={[1, 1.75]}
          gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        >
          <ambientLight intensity={0.35} />
          <directionalLight position={[6, 8, 10]} intensity={1.1} />
          <directionalLight position={[-8, -4, -6]} intensity={0.35} color="#4CE0E0" />
          <pointLight position={[0, 0, 6]} intensity={0.5} color="#4CE0E0" distance={22} />

          <CameraRig mode={mode} />
          <Starfield />
          <Molecule
            atoms={atoms}
            bonds={bonds}
            uncertainty={uncertainty}
            offset={MOLECULE_OFFSET[mode] ?? MOLECULE_OFFSET.idle}
          />
          {(mode === 'prediction' || mode === 'analyzer') && (
            <UncertaintyShell
              uncertainty={uncertainty}
              outOfDomain={outOfDomain}
              offset={MOLECULE_OFFSET[mode] ?? MOLECULE_OFFSET.idle}
            />
          )}
        </Canvas>
      </div>

      {/* Reading scrim. The molecule must never cost legibility, so the left
          two-thirds — where all the text lives — is darkened behind content. */}
      <div
        className="fixed inset-0 z-[1] pointer-events-none"
        style={{
          background:
            'linear-gradient(100deg, var(--in-base) 0%, color-mix(in srgb, var(--in-base) 88%, transparent) 42%, transparent 72%)',
        }}
      />

      <div className="relative z-10">{children}</div>
    </SceneContext.Provider>
  );
};

/**
 * Parse the backend's MOL block (from RDKit) into atoms and bonds.
 *
 * MOL V2000 format: line 4 is the counts line (atoms, bonds), then one line
 * per atom (x, y, z, element), then one line per bond (a, b, order) using
 * 1-based atom indices.
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
