import React, { useEffect, useState } from 'react';
import { motion, useSpring } from 'framer-motion';

export const CyberCursor: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [isPointer, setIsPointer] = useState(false);

  const mouseX = useSpring(0, { stiffness: 450, damping: 30 });
  const mouseY = useSpring(0, { stiffness: 450, damping: 30 });

  useEffect(() => {
    // Only enable on desktop/mouse devices
    if (window.matchMedia('(pointer: coarse)').matches) return;

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      if (!isVisible) setIsVisible(true);

      const target = e.target as HTMLElement | null;
      if (target) {
        const isClickable = 
          target.tagName === 'BUTTON' || 
          target.tagName === 'A' || 
          target.tagName === 'INPUT' || 
          target.closest('button') || 
          target.closest('a') ||
          target.getAttribute('role') === 'button';
        setIsPointer(!!isClickable);
      }
    };

    const handleMouseLeave = () => setIsVisible(false);

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [isVisible, mouseX, mouseY]);

  if (!isVisible) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      {/* Outer Glow Halo */}
      <motion.div
        className="fixed top-0 left-0 rounded-full bg-cyan-400/10 blur-xl pointer-events-none"
        style={{
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%',
          width: isPointer ? 120 : 70,
          height: isPointer ? 120 : 70,
          transition: 'width 0.2s, height 0.2s'
        }}
      />
      {/* Core Reticle Dot */}
      <motion.div
        className="fixed top-0 left-0 rounded-full border border-cyan-400/60 pointer-events-none flex items-center justify-center backdrop-blur-[1px]"
        style={{
          x: mouseX,
          y: mouseY,
          translateX: '-50%',
          translateY: '-50%',
          width: isPointer ? 32 : 14,
          height: isPointer ? 32 : 14,
          backgroundColor: isPointer ? 'rgba(0, 242, 254, 0.15)' : 'rgba(0, 242, 254, 0.4)',
          transition: 'width 0.18s, height 0.18s, background-color 0.18s'
        }}
      >
        {isPointer && (
          <div className="w-1.5 h-1.5 rounded-full bg-cyan-300 animate-ping" />
        )}
      </motion.div>
    </div>
  );
};

