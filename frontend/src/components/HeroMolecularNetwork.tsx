import React, { useEffect, useRef } from 'react';

interface AtomNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  symbol: string;
  color: string;
  glow: string;
  charge?: string;
  ringId?: number;
}

interface Bond {
  from: number;
  to: number;
  order: 1 | 2; // 1 = single bond, 2 = double bond
  aromatic?: boolean;
}

export const HeroMolecularNetwork: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 550);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 450);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    // Mouse coordinates
    const mouse = { x: -1000, y: -1000, radius: 100 };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      mouse.x = -1000;
      mouse.y = -1000;
    };

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    // Initialize Molecular Graph nodes
    // Build a stylized multi-ring polycyclic scaffold with side chains
    const nodes: AtomNode[] = [];
    const bonds: Bond[] = [];

    // Atom palettes (Scientific standard CPK/Jmol adapted for dark cyan theme)
    const ATOM_TYPES = [
      { symbol: 'C', color: '#06b6d4', glow: 'rgba(6, 182, 212, 0.4)', radius: 6 },
      { symbol: 'N', color: '#3b82f6', glow: 'rgba(59, 130, 246, 0.4)', radius: 6.5 },
      { symbol: 'O', color: '#f43f5e', glow: 'rgba(244, 63, 94, 0.4)', radius: 7 },
      { symbol: 'F', color: '#10b981', glow: 'rgba(16, 185, 129, 0.4)', radius: 6 },
      { symbol: 'S', color: '#eab308', glow: 'rgba(234, 179, 8, 0.4)', radius: 8 },
      { symbol: 'P', color: '#a855f7', glow: 'rgba(168, 85, 247, 0.4)', radius: 7.5 }
    ];

    // Seed 28 structured nodes representing a medicinal lead scaffold
    const centerX = width * 0.5;
    const centerY = height * 0.48;
    const scale = Math.min(width, height) * 0.38;

    // Benzene Ring 1
    const r1 = scale * 0.35;
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3 - Math.PI / 6;
      nodes.push({
        x: centerX - scale * 0.45 + r1 * Math.cos(angle),
        y: centerY + r1 * Math.sin(angle),
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        radius: 6,
        symbol: i === 1 ? 'N' : 'C',
        color: i === 1 ? '#3b82f6' : '#06b6d4',
        glow: i === 1 ? 'rgba(59, 130, 246, 0.4)' : 'rgba(6, 182, 212, 0.4)',
        ringId: 1
      });
      bonds.push({ from: i, to: (i + 1) % 6, order: i % 2 === 0 ? 2 : 1, aromatic: true });
    }

    // Benzene Ring 2 (Fused / Linked)
    const offset = 6;
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3;
      nodes.push({
        x: centerX + scale * 0.3 + r1 * Math.cos(angle),
        y: centerY + r1 * Math.sin(angle),
        vx: (Math.random() - 0.5) * 0.2,
        vy: (Math.random() - 0.5) * 0.2,
        radius: 6,
        symbol: i === 4 ? 'O' : (i === 2 ? 'N' : 'C'),
        color: i === 4 ? '#f43f5e' : (i === 2 ? '#3b82f6' : '#06b6d4'),
        glow: i === 4 ? 'rgba(244, 63, 94, 0.4)' : 'rgba(6, 182, 212, 0.4)',
        ringId: 2
      });
      bonds.push({ from: offset + i, to: offset + ((i + 1) % 6), order: i % 2 === 1 ? 2 : 1, aromatic: true });
    }

    // Bridge between Ring 1 and Ring 2 (Amide linkage: -C(=O)-NH-)
    nodes.push({
      x: centerX - scale * 0.1,
      y: centerY - scale * 0.18,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'C',
      color: '#06b6d4',
      glow: 'rgba(6, 182, 212, 0.4)',
      radius: 6.5
    }); // 12
    nodes.push({
      x: centerX - scale * 0.1,
      y: centerY - scale * 0.42,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: '=O',
      color: '#f43f5e',
      glow: 'rgba(244, 63, 94, 0.5)',
      radius: 7
    }); // 13 (Carbonyl)
    nodes.push({
      x: centerX + scale * 0.08,
      y: centerY - scale * 0.12,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'NH',
      color: '#3b82f6',
      glow: 'rgba(59, 130, 246, 0.5)',
      radius: 6.5
    }); // 14

    bonds.push({ from: 1, to: 12, order: 1 });
    bonds.push({ from: 12, to: 13, order: 2 });
    bonds.push({ from: 12, to: 14, order: 1 });
    bonds.push({ from: 14, to: offset + 3, order: 1 });

    // Functional sidechains
    // Hydroxyl -OH
    nodes.push({
      x: nodes[4].x - 45,
      y: nodes[4].y + 35,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'OH',
      color: '#f43f5e',
      glow: 'rgba(244, 63, 94, 0.4)',
      radius: 6
    }); // 15
    bonds.push({ from: 4, to: 15, order: 1 });

    // Fluorine -F
    nodes.push({
      x: nodes[offset + 0].x + 45,
      y: nodes[offset + 0].y - 30,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'F',
      color: '#10b981',
      glow: 'rgba(16, 185, 129, 0.4)',
      radius: 6
    }); // 16
    bonds.push({ from: offset + 0, to: 16, order: 1 });

    // Sulfonamide moiety -SO2NH2
    const sIndex = nodes.length;
    nodes.push({
      x: nodes[offset + 1].x + 40,
      y: nodes[offset + 1].y + 40,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'S',
      color: '#eab308',
      glow: 'rgba(234, 179, 8, 0.4)',
      radius: 8
    }); // 17
    nodes.push({
      x: nodes[offset + 1].x + 75,
      y: nodes[offset + 1].y + 20,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: '=O',
      color: '#f43f5e',
      glow: 'rgba(244, 63, 94, 0.4)',
      radius: 6
    }); // 18
    nodes.push({
      x: nodes[offset + 1].x + 55,
      y: nodes[offset + 1].y + 75,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      symbol: 'NH2',
      color: '#3b82f6',
      glow: 'rgba(59, 130, 246, 0.4)',
      radius: 6
    }); // 19

    bonds.push({ from: offset + 1, to: sIndex, order: 1 });
    bonds.push({ from: sIndex, to: sIndex + 1, order: 2 });
    bonds.push({ from: sIndex, to: sIndex + 2, order: 1 });

    // Orbiting/surrounding solvent cloud / molecular field points
    for (let i = 0; i < 8; i++) {
      const type = ATOM_TYPES[Math.floor(Math.random() * ATOM_TYPES.length)];
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        symbol: type.symbol,
        color: type.color,
        glow: type.glow,
        radius: type.radius * 0.75
      });
    }

    let pulse = 0;

    const render = () => {
      pulse += 0.03;
      ctx.clearRect(0, 0, width, height);

      // Subtle scientific background grid lines
      ctx.strokeStyle = 'rgba(30, 41, 59, 0.2)';
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Update node positions with gentle spring & boundary physics
      nodes.forEach((node) => {
        // Floating motion
        node.x += node.vx;
        node.y += node.vy;

        // Keep within canvas
        if (node.x < 20 || node.x > width - 20) node.vx *= -1;
        if (node.y < 20 || node.y > height - 20) node.vy *= -1;

        // Mouse interaction: gentle repulsion
        const dx = node.x - mouse.x;
        const dy = node.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < mouse.radius && dist > 0) {
          const force = (mouse.radius - dist) / mouse.radius;
          node.x += (dx / dist) * force * 3;
          node.y += (dy / dist) * force * 3;
        }
      });

      // Draw Chemical Bonds (lines)
      bonds.forEach((bond) => {
        const a = nodes[bond.from];
        const b = nodes[bond.to];
        if (!a || !b) return;

        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist === 0) return;

        // Perpendicular vector for double bonds
        const nx = -dy / dist;
        const ny = dx / dist;

        if (bond.order === 2) {
          // Double bond
          const spacing = 2.8;
          ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
          ctx.lineWidth = 1.8;

          ctx.beginPath();
          ctx.moveTo(a.x + nx * spacing, a.y + ny * spacing);
          ctx.lineTo(b.x + nx * spacing, b.y + ny * spacing);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(a.x - nx * spacing, a.y - ny * spacing);
          ctx.lineTo(b.x - nx * spacing, b.y - ny * spacing);
          ctx.stroke();
        } else {
          // Single bond
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
          ctx.lineWidth = 2;

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }

        // Animated orbital electron pulse traveling along bonds
        const t = (Math.sin(pulse + bond.from * 0.7) + 1) / 2;
        const px = a.x + dx * t;
        const py = a.y + dy * t;

        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw dynamic mouse interaction bonds if mouse is near nodes
      if (mouse.x > 0 && mouse.y > 0) {
        nodes.forEach((node) => {
          const dx = node.x - mouse.x;
          const dy = node.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 110) {
            ctx.strokeStyle = `rgba(6, 182, 212, ${0.4 * (1 - dist / 110)})`;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(node.x, node.y);
            ctx.lineTo(mouse.x, mouse.y);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        });
      }

      // Draw Atoms (Nodes)
      nodes.forEach((node) => {
        // Outer glow
        const glowRadius = node.radius * 2.2 + Math.sin(pulse + node.x) * 1.5;
        const gradient = ctx.createRadialGradient(node.x, node.y, node.radius * 0.5, node.x, node.y, glowRadius);
        gradient.addColorStop(0, node.glow);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        // Atom sphere
        ctx.fillStyle = node.color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fill();

        // High-tech specular inner highlight
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.beginPath();
        ctx.arc(node.x - node.radius * 0.3, node.y - node.radius * 0.3, node.radius * 0.35, 0, Math.PI * 2);
        ctx.fill();

        // Atom label text
        ctx.fillStyle = '#f8fafc';
        ctx.font = '600 10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (node.symbol.length > 1) {
          ctx.fillText(node.symbol, node.x + 14, node.y - 10);
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, []);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[420px] rounded-3xl overflow-hidden bg-slate-950/70 border border-slate-800/80 shadow-2xl backdrop-blur-xl group">
      {/* Background radial gradient glow */}
      <div className="absolute inset-0 bg-radial-gradient from-cyan-950/30 via-transparent to-transparent pointer-events-none" />

      {/* Canvas */}
      <canvas ref={canvasRef} className="w-full h-full cursor-crosshair" />

      {/* Overlay Scientific Telemetry Tags */}
      <div className="absolute top-4 left-4 pointer-events-none flex flex-col gap-1.5">
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/90 border border-cyan-500/30 text-[10px] text-cyan-300 font-mono tracking-wider backdrop-blur-md shadow-md">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>RDKit Graph Tensor (ECFP4 · 2048-bit)</span>
        </div>
        <div className="text-[10px] text-slate-400 font-mono pl-1 hidden sm:block">
          Conformer Energy: <span className="text-emerald-400 font-bold">-48.2 kcal/mol</span>
        </div>
      </div>

      <div className="absolute bottom-4 right-4 pointer-events-none text-right hidden sm:block">
        <div className="text-[10px] font-mono text-slate-400">
          Target: <span className="text-white font-semibold">JAK2 / EGFR Kinase Pocket</span>
        </div>
        <div className="text-[10px] font-mono text-cyan-400/90">
          BoTorch qUCB Acquisition: <span className="font-bold text-cyan-300">α(x) = 8.94</span>
        </div>
      </div>
    </div>
  );
};
