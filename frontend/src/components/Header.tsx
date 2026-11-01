import React from 'react';
import { 
  Atom, 
  FlaskConical, 
  Activity, 
  GitCompare, 
  Cpu, 
  BarChart2,
  BookOpen,
  ArrowUpRight
} from 'lucide-react';
import { motion } from 'framer-motion';

interface HeaderProps {
  activeTab: 'analyzer' | 'prediction' | 'pareto' | 'active_learning' | 'benchmark';
  setActiveTab: (tab: 'analyzer' | 'prediction' | 'pareto' | 'active_learning' | 'benchmark') => void;
  onReturnToLanding: () => void;
  onOpenGlossary: () => void;
  backendHealthy: boolean;
}

export const Header: React.FC<HeaderProps> = ({ 
  activeTab, 
  setActiveTab, 
  onReturnToLanding, 
  onOpenGlossary,
  backendHealthy 
}) => {
  const tabs = [
    { id: 'analyzer' as const, num: '01', label: 'Molecular Analyzer', shortLabel: 'Analyzer', icon: FlaskConical },
    { id: 'prediction' as const, num: '02', label: 'Prediction & Conformal', shortLabel: 'Prediction', icon: Activity },
    { id: 'pareto' as const, num: '03', label: 'Pareto Lab', shortLabel: 'Pareto', icon: GitCompare },
    { id: 'active_learning' as const, num: '04', label: 'Active Learning', shortLabel: 'Active ML', icon: Cpu },
    { id: 'benchmark' as const, num: '05', label: 'ESOL Benchmark', shortLabel: 'Benchmark', icon: BarChart2 }
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#060709]/80 backdrop-blur-2xl border-b border-white/[0.07] transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          
          {/* Brand Wordmark & Return Link */}
          <div className="flex items-center gap-4">
            <button
              onClick={onReturnToLanding}
              className="flex items-center gap-3 group cursor-pointer text-left focus:outline-none"
              title="Return to Overview"
            >
              <div className="w-9 h-9 rounded-full bg-white/[0.04] border border-white/[0.12] flex items-center justify-center group-hover:border-white/30 group-hover:bg-white/[0.08] transition-all duration-300">
                <Atom className="w-4 h-4 text-white group-hover:rotate-180 transition-transform duration-700" />
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span className="text-base font-medium tracking-tight text-white group-hover:text-slate-200 transition-colors">
                    MoleculeMind
                  </span>
                  <span className="text-[10px] tracking-wider uppercase font-mono px-2 py-0.5 rounded-full bg-white/[0.05] text-slate-300 border border-white/[0.08]">
                    v2.4
                  </span>
                </div>
                <span className="text-[10px] tracking-wider uppercase text-slate-400 font-mono">
                  Uncertainty-Aware Discovery
                </span>
              </div>
            </button>
          </div>

          {/* Floating Pill Navigation Tabs (Emotion Agency Dock Style) */}
          <nav className="flex items-center gap-1 bg-[#0d0e15]/90 p-1.5 rounded-full border border-white/[0.1] shadow-2xl overflow-x-auto max-w-full">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative px-3.5 sm:px-4 py-2 rounded-full text-xs font-medium transition-all duration-300 flex items-center gap-2 shrink-0 cursor-pointer focus:outline-none z-10 ${
                    isActive
                      ? 'text-zinc-950 font-semibold'
                      : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeTabPill"
                      className="absolute inset-0 bg-white rounded-full shadow-[0_2px_15px_rgba(255,255,255,0.25)] z-[-1]"
                      transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                    />
                  )}
                  <span className={`text-[10px] font-mono ${isActive ? 'text-zinc-500' : 'text-slate-400'}`}>
                    {tab.num}
                  </span>
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-zinc-950' : 'text-slate-400'}`} />
                  <span className="hidden sm:inline">{tab.shortLabel}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Action Menu: Status & Scientific Guide */}
          <div className="flex items-center gap-3">
            {/* Backend Connectivity Status Dot */}
            <div 
              className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.08] text-[11px] font-mono text-slate-300"
              title={backendHealthy ? "Delaney ESOL API Online" : "Connecting to Backend"}
            >
              <span className={`w-2 h-2 rounded-full ${backendHealthy ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-amber-400 animate-pulse'}`} />
              <span className="text-slate-400">{backendHealthy ? 'API Online' : 'Connecting'}</span>
            </div>

            {/* Scientific Theory & Guide Button */}
            <button
              onClick={onOpenGlossary}
              className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium text-slate-200 bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.12] hover:border-white/25 transition-all cursor-pointer"
            >
              <BookOpen className="w-3.5 h-3.5 text-slate-300" />
              <span className="hidden md:inline">Scientific Guide</span>
            </button>

            {/* Return to Overview Button */}
            <button
              onClick={onReturnToLanding}
              className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-medium text-zinc-950 bg-white hover:bg-slate-200 transition-all shadow-md cursor-pointer"
            >
              <span>Overview</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>

        </div>
      </div>
    </header>
  );
};
