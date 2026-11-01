import React, { useEffect, useState } from 'react';
import { motion, useSpring } from 'framer-motion';

export const AgencyCursor: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [isPointer, setIsPointer] = useState(false);

  const cursorX = useSpring(0, { stiffness: 500, damping: 32 });
  const cursorY = useSpring(0, { stiffness: 500, damping: 32 });

  useEffect(() => {
    // Only enable on non-touch devices with fine pointers
    if (window.matchMedia('(pointer: coarse)').matches) return;

    const handleMouseMove = (e: MouseEvent) => {
      cursorX.set(e.clientX);
      cursorY.set(e.clientY);
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
  }, [isVisible, cursorX, cursorY]);

  if (!isVisible) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      {/* Smooth Trailing Ambient Ring */}
      <motion.div
        className="fixed top-0 left-0 rounded-full border border-white/20 bg-white/[0.03] backdrop-blur-[2px] pointer-events-none transition-all duration-200"
        style={{
          x: cursorX,
          y: cursorY,
          translateX: '-50%',
          translateY: '-50%',
          width: isPointer ? 48 : 28,
          height: isPointer ? 48 : 28,
          borderColor: isPointer ? 'rgba(255, 255, 255, 0.45)' : 'rgba(255, 255, 255, 0.2)',
          boxShadow: isPointer ? '0 0 20px rgba(255, 255, 255, 0.15)' : 'none'
        }}
      />
      {/* Center Micro Dot */}
      <motion.div
        className="fixed top-0 left-0 w-1.5 h-1.5 rounded-full bg-white pointer-events-none"
        style={{
          x: cursorX,
          y: cursorY,
          translateX: '-50%',
          translateY: '-50%',
          scale: isPointer ? 0.6 : 1
        }}
      />
    </div>
  );
};

