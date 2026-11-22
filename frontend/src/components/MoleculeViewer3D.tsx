import React, { useEffect, useRef, useState } from 'react';
import { RotateCw, RefreshCw, Sparkles, AlertTriangle } from 'lucide-react';

interface MoleculeViewer3DProps {
  sdfData?: string | null;
  moleculeName?: string;
  /** Backend-reported reason the conformer could not be generated, if any. */
  sdfError?: string | null;
}

// Wait for the 3Dmol bundle to appear on window. It is a classic <script> tag,
// so it may not have executed by the time this component first renders.
const waitFor3Dmol = (timeoutMs = 8000): Promise<boolean> =>
  new Promise((resolve) => {
    if (window.$3Dmol) return resolve(true);
    const started = Date.now();
    const timer = window.setInterval(() => {
      if (window.$3Dmol) {
        window.clearInterval(timer);
        resolve(true);
      } else if (Date.now() - started > timeoutMs) {
        window.clearInterval(timer);
        resolve(false);
      }
    }, 100);
  });

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export const MoleculeViewer3D: React.FC<MoleculeViewer3DProps> = ({ sdfData, moleculeName, sdfError }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [styleMode, setStyleMode] = useState<'stick' | 'sphere' | 'ballStick'>('ballStick');
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [hasSurface, setHasSurface] = useState<boolean>(false);
  const [surfaceObj, setSurfaceObj] = useState<any>(null);
  // The viewer previously failed to a blank box whenever the 3Dmol bundle was
  // unavailable; these states make every failure mode visible and recoverable.
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState<number>(0);

  useEffect(() => {
    if (!sdfData) {
      setStatus('idle');
      return;
    }

    let cancelled = false;
    setStatus('loading');
    setErrorMessage(null);

    waitFor3Dmol().then((available) => {
      if (cancelled) return;

      if (!available) {
        setStatus('error');
        setErrorMessage(
          'The 3Dmol.js viewer library did not load. It is served locally from ' +
          '/3Dmol-min.js, so this usually means the asset is missing from the build ' +
          'output or was blocked by the browser.'
        );
        return;
      }
      if (!containerRef.current) return;

      try {
        containerRef.current.innerHTML = '';
        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: '#030712'
        });
        viewerRef.current = viewer;

        viewer.addModel(sdfData, 'sdf');
        applyStyle(viewer, styleMode);
        viewer.zoomTo();
        viewer.render();
        if (isRotating) viewer.spin('y', 1.0);

        setStatus('ready');
      } catch (err) {
        setStatus('error');
        setErrorMessage(
          err instanceof Error
            ? `The conformer could not be rendered: ${err.message}`
            : 'The conformer could not be rendered.'
        );
      }
    });

    return () => {
      cancelled = true;
      if (viewerRef.current) {
        try {
          viewerRef.current.clear();
        } catch {
          /* viewer already torn down */
        }
      }
    };
  }, [sdfData, reloadToken]);

  const applyStyle = (viewer: any, mode: 'stick' | 'sphere' | 'ballStick') => {
    if (!viewer) return;
    viewer.setStyle({}, {}); // Reset

    if (mode === 'stick') {
      viewer.setStyle({}, { stick: { radius: 0.2, colorscheme: 'Jmol' } });
    } else if (mode === 'sphere') {
      viewer.setStyle({}, { sphere: { scale: 0.8, colorscheme: 'Jmol' } });
    } else {
      // Ball and Stick
      viewer.setStyle({}, {
        stick: { radius: 0.14, colorscheme: 'Jmol' },
        sphere: { scale: 0.28, colorscheme: 'Jmol' }
      });
    }
    viewer.render();
  };

  const handleStyleChange = (mode: 'stick' | 'sphere' | 'ballStick') => {
    setStyleMode(mode);
    if (viewerRef.current) {
      applyStyle(viewerRef.current, mode);
    }
  };

  const toggleRotation = () => {
    if (!viewerRef.current) return;
    if (isRotating) {
      viewerRef.current.spin(false);
      setIsRotating(false);
    } else {
      viewerRef.current.spin('y', 1.0);
      setIsRotating(true);
    }
  };

  const toggleSurface = () => {
    if (!viewerRef.current) return;
    if (hasSurface) {
      if (surfaceObj) {
        viewerRef.current.removeSurface(surfaceObj);
        setSurfaceObj(null);
      }
      setHasSurface(false);
    } else {
      try {
        const surf = viewerRef.current.addSurface(
          window.$3Dmol.SurfaceType.VDW,
          { opacity: 0.65, color: '#06b6d4' }
        );
        setSurfaceObj(surf);
        setHasSurface(true);
      } catch (e) {
        console.error("Could not add surface", e);
      }
    }
    viewerRef.current.render();
  };

  const resetView = () => {
    if (!viewerRef.current) return;
    viewerRef.current.zoomTo();
    viewerRef.current.render();
  };

  // Empty state: nothing requested yet, or the backend could not embed a conformer.
  if (!sdfData) {
    if (sdfError) {
      return (
        <div className="flex flex-col items-center justify-center h-72 rounded-xl bg-amber-500/5 border border-amber-500/20 text-amber-200 p-6 text-center gap-2">
          <AlertTriangle className="w-7 h-7 text-amber-400" />
          <p className="text-sm font-medium">No 3D conformer could be generated</p>
          <p className="text-xs text-amber-200/70 max-w-sm">{sdfError}</p>
          <p className="text-[11px] text-slate-400 mt-1">
            The 2D structure and all predicted properties are unaffected.
          </p>
        </div>
      );
    }
    return (
      <div className="flex flex-col items-center justify-center h-72 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-400 p-6 text-center">
        <RotateCw className="w-8 h-8 text-slate-600 mb-2 animate-spin-slow" />
        <p className="text-sm">Enter a valid SMILES to generate 3D conformer coordinates.</p>
      </div>
    );
  }

  // Error state: the viewer library or the render itself failed.
  if (status === 'error') {
    return (
      <div className="flex flex-col items-center justify-center h-72 rounded-xl bg-rose-500/5 border border-rose-500/20 text-rose-200 p-6 text-center gap-3">
        <AlertTriangle className="w-7 h-7 text-rose-400" />
        <p className="text-sm font-medium">3D viewer unavailable</p>
        <p className="text-xs text-rose-200/70 max-w-sm">{errorMessage}</p>
        <button
          onClick={() => setReloadToken((t) => t + 1)}
          className="mt-1 px-4 py-1.5 text-xs font-medium rounded-full bg-white/10 hover:bg-white/20 text-white border border-white/20 transition-colors cursor-pointer"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-800/80 bg-slate-950 shadow-inner group">
      {/* 3Dmol.js Canvas Container */}
      <div 
        ref={containerRef} 
        className="w-full h-80 sm:h-96 cursor-grab active:cursor-grabbing" 
      />

      {/* Floating Toolbar Controls */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-slate-900/85 backdrop-blur-md border border-slate-700/60 p-1.5 rounded-lg shadow-lg z-10">
        <button
          onClick={() => handleStyleChange('ballStick')}
          className={`px-2 py-1 text-xs font-medium rounded transition-all ${
            styleMode === 'ballStick' 
              ? 'bg-cyan-500 text-slate-950 font-semibold shadow' 
              : 'text-slate-300 hover:text-white hover:bg-slate-800'
          }`}
          title="Ball & Stick Mode"
        >
          Ball & Stick
        </button>
        <button
          onClick={() => handleStyleChange('stick')}
          className={`px-2 py-1 text-xs font-medium rounded transition-all ${
            styleMode === 'stick' 
              ? 'bg-cyan-500 text-slate-950 font-semibold shadow' 
              : 'text-slate-300 hover:text-white hover:bg-slate-800'
          }`}
          title="Stick Mode"
        >
          Stick
        </button>
        <button
          onClick={() => handleStyleChange('sphere')}
          className={`px-2 py-1 text-xs font-medium rounded transition-all ${
            styleMode === 'sphere' 
              ? 'bg-cyan-500 text-slate-950 font-semibold shadow' 
              : 'text-slate-300 hover:text-white hover:bg-slate-800'
          }`}
          title="Space-filling Sphere Mode"
        >
          Spacefill
        </button>

        <div className="w-[1px] h-4 bg-slate-700 mx-1" />

        <button
          onClick={toggleSurface}
          className={`p-1.5 rounded text-xs transition-all ${
            hasSurface ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40' : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
          title="Toggle VDW Surface"
        >
          <Sparkles className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={toggleRotation}
          className={`p-1.5 rounded text-xs transition-all ${
            isRotating ? 'text-cyan-400 bg-slate-800' : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
          title={isRotating ? "Pause Spin" : "Auto Spin"}
        >
          <RotateCw className={`w-3.5 h-3.5 ${isRotating ? 'animate-spin' : ''}`} />
        </button>

        <button
          onClick={resetView}
          className="p-1.5 rounded text-xs text-slate-400 hover:text-white hover:bg-slate-800 transition-all"
          title="Reset Zoom & Center"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Footer watermark & hint */}
      <div className="absolute bottom-2 left-3 flex items-center gap-2 text-[11px] text-slate-400 bg-slate-900/60 backdrop-blur-sm px-2.5 py-1 rounded-md border border-slate-800 pointer-events-none">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span>3Dmol.js MMFF94 Optimized Conformer</span>
        {moleculeName && <span className="text-slate-300 font-medium">({moleculeName})</span>}
      </div>
    </div>
  );
};
