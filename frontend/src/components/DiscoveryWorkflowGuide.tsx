import React from 'react';
import { 
  FlaskConical, 
  Activity, 
  ShieldCheck, 
  GitCompare, 
  Cpu, 
  Sparkles
} from 'lucide-react';

interface WorkflowGuideProps {
  currentTab: 'analyzer' | 'prediction' | 'pareto' | 'discovery' | 'active_learning' | 'benchmark';
  onNavigateTab: (tab: 'analyzer' | 'prediction' | 'pareto' | 'discovery' | 'active_learning' | 'benchmark') => void;
  activeSmiles?: string;
}

export const DiscoveryWorkflowGuide: React.FC<WorkflowGuideProps> = ({
  currentTab,
  onNavigateTab,
  activeSmiles
}) => {
  const steps = [
    {
      id: 'analyzer' as const,
      num: '01',
      title: 'Structure Validation',
      shortDesc: 'SMILES & 3D Conformer',
      icon: FlaskConical,
      tab: 'analyzer' as const
    },
    {
      id: 'prediction' as const,
      num: '02',
      title: 'Property Prediction',
      shortDesc: 'Delaney ESOL Regression',
      icon: Activity,
      tab: 'prediction' as const
    },
    {
      id: 'uncertainty' as const,
      num: '03',
      title: 'Conformal Intervals',
      shortDesc: 'Calibrated Bounds (1-α)',
      icon: ShieldCheck,
      tab: 'prediction' as const
    },
    {
      id: 'pareto' as const,
      num: '04',
      title: 'Pareto Laboratory',
      shortDesc: 'Multi-Objective Trade-Offs',
      icon: GitCompare,
      tab: 'pareto' as const
    },
    {
      id: 'active_learning' as const,
      num: '05',
      title: 'Active Learning',
      shortDesc: 'BoTorch GP Acquisition',
      icon: Cpu,
      tab: 'active_learning' as const
    }
  ];

  const getStepStatus = (stepTab: string, stepNum: string) => {
    if (currentTab === stepTab) return 'active';
    if (stepNum === '03' && currentTab === 'prediction') return 'active';
    return 'inactive';
  };

  return (
    <div className="mb-8 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] p-4 sm:p-5 shadow-2xl backdrop-blur-2xl">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 mb-4 pb-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2.5">
          <Sparkles className="w-4 h-4 text-slate-300" />
          <span className="text-xs font-semibold text-white tracking-wider uppercase">
            Discovery Workflow Pipeline
          </span>
          <span className="text-slate-600 text-xs hidden sm:inline">•</span>
          <span className="text-xs text-slate-400 hidden sm:inline">
            Follow the 5-stage sequential decision cycle from molecular parsing to active learning
          </span>
        </div>

        {activeSmiles && (
          <div className="text-[11px] font-mono text-slate-400 flex items-center gap-2 truncate max-w-md bg-black/40 px-3 py-1 rounded-full border border-white/[0.07]">
            <span className="text-slate-500">Active:</span>
            <span className="text-slate-200 truncate">{activeSmiles}</span>
          </div>
        )}
      </div>

      {/* Steps Pipeline Horizontal Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {steps.map((step) => {
          const Icon = step.icon;
          const status = getStepStatus(step.tab, step.num);
          const isActive = status === 'active';

          return (
            <button
              key={step.num}
              onClick={() => onNavigateTab(step.tab)}
              className={`text-left p-3.5 rounded-2xl border transition-all cursor-pointer relative overflow-hidden group ${
                isActive
                  ? 'bg-white/[0.08] border-white/30 shadow-[0_4px_20px_rgba(255,255,255,0.08)]'
                  : 'bg-white/[0.02] border-white/[0.06] hover:border-white/15 hover:bg-white/[0.04]'
              }`}
            >
              {isActive && (
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-white" />
              )}
              <div className="flex items-center justify-between mb-2">
                <span className={`text-[10px] font-mono font-medium tracking-wider ${isActive ? 'text-white' : 'text-slate-500'}`}>
                  STAGE {step.num}
                </span>
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-500 group-hover:text-slate-300'}`} />
              </div>

              <div className={`text-xs font-medium truncate ${isActive ? 'text-white font-semibold' : 'text-slate-300 group-hover:text-white'}`}>
                {step.title}
              </div>

              <div className="text-[10px] text-slate-400 truncate mt-0.5 font-mono">
                {step.shortDesc}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
