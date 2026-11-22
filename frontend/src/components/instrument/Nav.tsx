/**
 * Nav — the module index.
 *
 * Modules are presented as a numbered index rather than buttons. The active
 * entry expands to show its subtitle; the others collapse to a number and
 * name. A single shared underline slides between them (Framer's `layoutId`),
 * which is what makes switching feel like one continuous object moving rather
 * than two separate states.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { EASE, DUR, Magnetic } from './primitives';
import type { SceneMode } from './SceneCanvas';

export interface ModuleDef {
  id: SceneMode;
  num: string;
  label: string;
  subtitle: string;
}

export const MODULES: ModuleDef[] = [
  { id: 'analyzer',        num: '01', label: 'Analyzer',     subtitle: 'Structure and descriptors' },
  { id: 'prediction',      num: '02', label: 'Prediction',   subtitle: 'Properties with calibrated bounds' },
  { id: 'pareto',          num: '03', label: 'Pareto',       subtitle: 'Multi-objective trade-offs' },
  { id: 'discovery',       num: '04', label: 'Discovery',    subtitle: 'Probability, risk, budget, generation' },
  { id: 'active_learning', num: '05', label: 'Campaign',     subtitle: 'Choosing the next experiment' },
  { id: 'benchmark',       num: '06', label: 'Benchmark',    subtitle: 'Measured performance' },
];

interface NavProps {
  active: SceneMode;
  onChange: (id: SceneMode) => void;
  backendHealthy: boolean;
  onHome: () => void;
}

export const Nav: React.FC<NavProps> = ({ active, onChange, backendHealthy, onHome }) => (
  <header
    className="sticky top-0 z-30"
    style={{
      // Solid, not translucent. backdrop-filter over scrolling content was
      // the single largest frame cost in the previous theme.
      background: 'var(--pp-paper)',
      borderBottom: '1px solid var(--pp-rule)',
    }}
  >
    <div className="max-w-[1400px] mx-auto px-5 sm:px-8">
      <div className="flex items-center justify-between h-16">
        {/* Wordmark */}
        <Magnetic strength={3}>
          <button
            onClick={onHome}
            className="in-display text-sm tracking-tight cursor-pointer transition-opacity hover:opacity-60"
            style={{ letterSpacing: '-0.02em' }}
          >
            MoleculeMind
            <span className="in-num ml-2 text-[9px]" style={{ color: 'var(--in-faint)' }}>
              v2
            </span>
          </button>
        </Magnetic>

        {/* Status — a quiet instrument light, not a banner */}
        <div className="flex items-center gap-2">
          <motion.span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: backendHealthy ? 'var(--in-signal)' : 'var(--in-caution)' }}
            animate={backendHealthy ? { opacity: [0.4, 1, 0.4] } : { opacity: 1 }}
            transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
          />
          <span className="in-label">{backendHealthy ? 'online' : 'connecting'}</span>
        </div>
      </div>

      {/* Module index */}
      <nav className="flex items-stretch gap-1 overflow-x-auto no-scrollbar -mb-px">
        {MODULES.map((m) => {
          const isActive = m.id === active;
          return (
            <button
              key={m.id}
              onClick={() => onChange(m.id)}
              className="relative shrink-0 text-left px-4 py-3 cursor-pointer group"
              style={{ minWidth: isActive ? 220 : 128 }}
            >
              <motion.div
                layout
                transition={{ duration: DUR.base, ease: EASE }}
                className="flex items-baseline gap-2"
              >
                <span
                  className="in-num text-[10px]"
                  style={{ color: isActive ? 'var(--in-signal)' : 'var(--in-faint)' }}
                >
                  {m.num}
                </span>
                <span
                  className="in-display text-[13px] transition-colors"
                  style={{ color: isActive ? 'var(--in-text)' : 'var(--in-dim)' }}
                >
                  {m.label}
                </span>
              </motion.div>

              {/* The subtitle only exists for the active module. */}
              <motion.div
                initial={false}
                animate={{
                  height: isActive ? 'auto' : 0,
                  opacity: isActive ? 1 : 0,
                }}
                transition={{ duration: DUR.fast, ease: EASE }}
                className="overflow-hidden"
              >
                <span className="in-label block mt-1 whitespace-nowrap">{m.subtitle}</span>
              </motion.div>

              {/* One shared underline that slides between modules. */}
              {isActive && (
                <motion.div
                  layoutId="nav-underline"
                  className="absolute left-4 right-4 bottom-0 h-[2px]"
                  style={{ background: 'var(--in-signal)' }}
                  transition={{ duration: DUR.base, ease: EASE }}
                />
              )}
            </button>
          );
        })}
      </nav>
    </div>
  </header>
);
