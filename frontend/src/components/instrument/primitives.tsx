/**
 * Instrument — motion and layout primitives.
 *
 * Everything in the app composes from these, which is what makes the motion
 * feel like one system rather than a collection of effects. One easing curve,
 * three duration tiers, one stagger interval.
 */

import React, { useEffect, useRef, useState } from 'react';
import { motion, useInView, useMotionValue, useSpring, useTransform } from 'framer-motion';

/** The single easing curve. Slow out, snappy in. */
export const EASE = [0.16, 1, 0.3, 1] as const;
export const DUR = { fast: 0.4, base: 0.7, cine: 1.2 } as const;
export const STAGGER = 0.04;

/** Fade and rise. The default entrance for everything. */
export const Rise: React.FC<{
  children: React.ReactNode;
  delay?: number;
  y?: number;
  className?: string;
}> = ({ children, delay = 0, y = 16, className }) => (
  <motion.div
    initial={{ opacity: 0, y }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: DUR.base, ease: EASE, delay }}
    className={className}
  >
    {children}
  </motion.div>
);

/** Staggers its children. Index-based so it works with mapped lists. */
export const Stagger: React.FC<{
  children: React.ReactNode;
  className?: string;
  delay?: number;
}> = ({ children, className, delay = 0 }) => (
  <motion.div
    className={className}
    initial="hidden"
    animate="shown"
    variants={{
      hidden: {},
      shown: { transition: { staggerChildren: STAGGER, delayChildren: delay } },
    }}
  >
    {children}
  </motion.div>
);

export const StaggerItem: React.FC<{ children: React.ReactNode; className?: string }> = ({
  children,
  className,
}) => (
  <motion.div
    className={className}
    variants={{
      hidden: { opacity: 0, y: 14 },
      shown: { opacity: 1, y: 0, transition: { duration: DUR.base, ease: EASE } },
    }}
  >
    {children}
  </motion.div>
);

/**
 * A heading that reveals word by word from behind a mask.
 *
 * This is the one piece of "showy" typography in the system, used only on
 * module titles. Each word sits in an overflow-hidden box and slides up, which
 * reads as a reveal rather than a fade.
 */
export const RevealText: React.FC<{
  text: string;
  className?: string;
  delay?: number;
}> = ({ text, className, delay = 0 }) => (
  <span className={className} style={{ display: 'inline-flex', flexWrap: 'wrap', gap: '0.28em' }}>
    {text.split(' ').map((word, i) => (
      <span key={`${word}-${i}`} style={{ overflow: 'hidden', display: 'inline-block' }}>
        <motion.span
          style={{ display: 'inline-block' }}
          initial={{ y: '110%' }}
          animate={{ y: 0 }}
          transition={{ duration: DUR.cine, ease: EASE, delay: delay + i * 0.055 }}
        >
          {word}
        </motion.span>
      </span>
    ))}
  </span>
);

/**
 * A number that counts to its value, in monospace.
 *
 * Animating the number rather than the container is the difference between
 * motion that decorates and motion that carries information — you watch the
 * value arrive.
 */
export const CountUp: React.FC<{
  value: number;
  decimals?: number;
  className?: string;
  suffix?: string;
}> = ({ value, decimals = 3, className, suffix = '' }) => {
  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 70, damping: 18 });
  const text = useTransform(spring, (v) => v.toFixed(decimals) + suffix);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });

  useEffect(() => {
    if (inView) mv.set(value);
  }, [inView, value, mv]);

  return (
    <span ref={ref} className={className}>
      <motion.span>{text}</motion.span>
    </span>
  );
};

/**
 * Wraps any value whose uncertainty we know and blurs it accordingly.
 *
 * This is the system's signature rule expressed as a component: a prediction
 * the models are unsure about is literally harder to read. `sigma` is the real
 * conformal scale; `scale` is the sigma at which we consider it fully unsure.
 */
/**
 * Map a 90% conformal half-width onto the 0..1 scalar the visual rules use.
 *
 * Shared so the figure and the numbers agree, and so the floor below which a
 * prediction counts as confident is stated in exactly one place.
 */
export const uncertaintyScalar = (
  marginError: number,
  floor = 1.0,
  scale = 2.4
) => Math.min(Math.max((marginError - floor) / scale, 0), 1);

export const Uncertain: React.FC<{
  children: React.ReactNode;
  sigma: number;
  scale?: number;
  /** Interval half-width below which a prediction is treated as confident. */
  floor?: number;
  outOfDomain?: boolean;
  className?: string;
}> = ({
  children,
  sigma,
  scale = 2.4,
  floor = 1.0,
  outOfDomain = false,
  className,
}) => {
  // Mapping sigma straight to blur from zero was wrong in practice. The 90%
  // half-width for a molecule the model handles well is already around 1.1
  // log units, so even the best predictions came out visibly soft and read as
  // a rendering defect rather than as a statement. The scale now starts at
  // that floor, so "well predicted" is sharp and the blur is reserved for
  // intervals that are genuinely wide.
  const u = Math.min(Math.max((sigma - floor) / scale, 0), 1);
  return (
    <span
      className={`${outOfDomain ? 'in-out-of-domain' : 'in-uncertain'} ${className ?? ''}`}
      style={{ ['--u' as string]: u.toFixed(3), display: 'inline-block' }}
      title={
        outOfDomain
          ? 'Outside the applicability domain — rendered hazy because the models cannot be trusted here.'
          : `Blur grows with the calibrated interval half-width (${sigma.toFixed(2)} log units); a confident prediction stays sharp.`
      }
    >
      {children}
    </span>
  );
};

/**
 * Proximity-magnetic hover: the element leans toward the cursor.
 *
 * Kept deliberately small (6px) — enough to feel alive, not enough to make a
 * button hard to hit.
 */
export const Magnetic: React.FC<{
  children: React.ReactNode;
  strength?: number;
  className?: string;
}> = ({ children, strength = 6, className }) => {
  const ref = useRef<HTMLDivElement>(null);
  const x = useSpring(0, { stiffness: 180, damping: 16 });
  const y = useSpring(0, { stiffness: 180, damping: 16 });

  return (
    <motion.div
      ref={ref}
      className={className}
      style={{ x, y }}
      onPointerMove={(e) => {
        const el = ref.current;
        if (!el) return;
        const r = el.getBoundingClientRect();
        x.set(((e.clientX - (r.left + r.width / 2)) / (r.width / 2)) * strength);
        y.set(((e.clientY - (r.top + r.height / 2)) / (r.height / 2)) * strength);
      }}
      onPointerLeave={() => {
        x.set(0);
        y.set(0);
      }}
    >
      {children}
    </motion.div>
  );
};

/** A labelled readout: monospace value over a tracked uppercase label. */
export const Readout: React.FC<{
  label: string;
  value: React.ReactNode;
  unit?: string;
  accent?: boolean;
  className?: string;
}> = ({ label, value, unit, accent, className }) => (
  <div className={`in-panel in-panel-hover p-5 ${className ?? ''}`}>
    <div className="in-label">{label}</div>
    <div
      className="in-num text-2xl font-light mt-2"
      style={{ color: accent ? 'var(--in-signal)' : 'var(--in-text)' }}
    >
      {value}
      {unit && <span className="text-[11px] ml-1.5" style={{ color: 'var(--in-faint)' }}>{unit}</span>}
    </div>
  </div>
);

/**
 * The boot sequence. Counts real progress while the backend trains or loads,
 * because a fake progress bar is worse than none.
 */
export const BootSequence: React.FC<{ onDone: () => void; ready: boolean }> = ({
  onDone,
  ready,
}) => {
  const [pct, setPct] = useState(0);

  useEffect(() => {
    // Climb to 90% on a curve, then wait for the backend to actually be ready.
    const id = window.setInterval(() => {
      setPct((p) => (p < 90 ? p + (90 - p) * 0.08 + 0.4 : p));
    }, 40);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!ready) return;
    setPct(100);
    const id = window.setTimeout(onDone, 700);
    return () => window.clearTimeout(id);
  }, [ready, onDone]);

  return (
    <motion.div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center"
      style={{ background: 'var(--in-base)' }}
      exit={{ opacity: 0 }}
      transition={{ duration: DUR.cine, ease: EASE }}
    >
      <div className="in-display-xl text-[13vw] sm:text-[8vw] leading-none">
        <RevealText text="MOLECULEMIND" />
      </div>
      <div className="w-[min(420px,70vw)] mt-10">
        <div className="h-[1px] w-full" style={{ background: 'var(--in-line)' }}>
          <motion.div
            className="h-full"
            style={{ background: 'var(--in-signal)' }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.3, ease: 'linear' }}
          />
        </div>
        <div className="flex justify-between mt-3">
          <span className="in-label">
            {ready ? 'models ready' : 'loading models'}
          </span>
          <span className="in-num text-[10px]" style={{ color: 'var(--in-signal)' }}>
            {Math.round(pct).toString().padStart(3, '0')}
          </span>
        </div>
      </div>
    </motion.div>
  );
};
