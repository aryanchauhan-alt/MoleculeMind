import React, { useState } from 'react';
import { 
  Atom, 
  FlaskConical, 
  Activity, 
  ShieldCheck, 
  GitCompare, 
  Cpu, 
  ArrowRight, 
  ArrowUpRight, 
  Sparkles, 
  BookOpen, 
  ChevronRight
} from 'lucide-react';
import { motion } from 'framer-motion';
import confetti from 'canvas-confetti';
import { QuantumMolecularOrb } from './QuantumMolecularOrb';

interface LandingPageProps {
  onLaunchApp: (tab?: 'analyzer' | 'prediction' | 'pareto' | 'active_learning' | 'benchmark') => void;
  onOpenGlossary: () => void;
  backendHealthy: boolean;
}

const PRESET_SHOWCASE = [
  { 
    name: 'Aspirin', 
    formula: 'C9H8O4',
    smiles: 'CC(=O)Oc1ccccc1C(=O)O', 
    logS: -1.61, 
    conf90: '[-2.12, -1.09]', 
    domain: 'IN_DOMAIN',
    desc: 'Classic non-steroidal anti-inflammatory compound well within Delaney training chemical space.',
    badgeClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
  },
  { 
    name: 'Caffeine', 
    formula: 'C8H10N4O2',
    smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C', 
    logS: -1.02, 
    conf90: '[-1.53, -0.51]', 
    domain: 'IN_DOMAIN',
    desc: 'Central nervous system stimulant with high aqueous solubility and tight conformal interval.',
    badgeClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
  },
  { 
    name: 'Ibuprofen', 
    formula: 'C13H18O2',
    smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 
    logS: -3.17, 
    conf90: '[-3.68, -2.66]', 
    domain: 'IN_DOMAIN',
    desc: 'Moderately lipophilic analgesic with balanced solubility and verified applicability domain.',
    badgeClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
  },
  { 
    name: 'Diclofenac', 
    formula: 'C14H11Cl2NO2',
    smiles: 'O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl', 
    logS: -4.80, 
    conf90: '[-5.31, -4.29]', 
    domain: 'BORDERLINE',
    desc: 'Dichlorinated phenylacetic acid derivative exhibiting elevated statistical leverage.',
    badgeClass: 'text-amber-400 bg-amber-500/10 border-amber-500/20'
  },
  { 
    name: 'Imatinib core', 
    formula: 'C29H31N7O',
    smiles: 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1', 
    logS: -5.10, 
    conf90: '[-5.61, -4.59]', 
    domain: 'OUT_OF_DOMAIN',
    desc: 'Synthetic tyrosine kinase inhibitor scaffold structurally remote from simple aqueous training space.',
    badgeClass: 'text-rose-400 bg-rose-500/10 border-rose-500/20'
  }
];

export const LandingPage: React.FC<LandingPageProps> = ({ 
  onLaunchApp, 
  onOpenGlossary,
  backendHealthy 
}) => {
  const [selectedDemo, setSelectedDemo] = useState(PRESET_SHOWCASE[0]);

  const handleLaunchWithConfetti = (tab?: 'analyzer' | 'prediction' | 'pareto' | 'active_learning' | 'benchmark') => {
    confetti({
      particleCount: 50,
      spread: 60,
      origin: { y: 0.8 },
      colors: ['#ffffff', '#38bdf8', '#34d399']
    });
    onLaunchApp(tab);
  };

  const capabilities = [
    {
      id: 'analyzer' as const,
      num: '01',
      title: 'Molecular Analyzer',
      subtitle: 'Structure Validation & 3D Minimization',
      badge: 'RDKit & 3Dmol.js',
      description: 'Instant SMILES parsing, publication-grade 2D vector depictions, and MMFF94 force-field 3D geometry generation.',
      details: 'Calculates Lipinski Rule of 5 oral bioavailability compliance, TPSA, and QED drug-likeness indices in sub-millisecond execution cycles.',
      icon: FlaskConical
    },
    {
      id: 'prediction' as const,
      num: '02',
      title: 'Property Prediction',
      subtitle: 'Delaney ESOL Trained Architecture',
      badge: 'ExtraTrees Regressor',
      description: 'Genuine multi-feature regression model trained on 1,128 authentic Delaney ESOL measured compounds.',
      details: 'Evaluated on held-out test data: R² = 0.894, RMSE = 0.704 log(mol/L), MAE = 0.482. Backed by 1,024-bit Morgan Fingerprints.',
      icon: Activity
    },
    {
      id: 'prediction' as const,
      num: '03',
      title: 'Conformal Uncertainty',
      subtitle: 'Distribution-Free Coverage',
      badge: 'Inductive Conformal Prediction',
      description: 'Finite-sample confidence intervals guaranteeing nominal coverage P(Y ∈ [L, U]) ≥ 1 - α without Gaussian assumptions.',
      details: 'Replaces dangerous point estimates with calibrated intervals at 80%, 90%, and 95% confidence based on nonconformity residuals.',
      icon: ShieldCheck
    },
    {
      id: 'prediction' as const,
      num: '04',
      title: 'Applicability Domain',
      subtitle: 'OECD Guidance Document No. 69',
      badge: 'Tanimoto & PCA Leverage',
      description: 'Evaluates structural proximity to the training space using Morgan fingerprint Tanimoto distance and PCA statistical leverage.',
      details: 'Flags molecules as IN_DOMAIN, BORDERLINE, or OUT_OF_DOMAIN to prevent catastrophic extrapolation on unfamiliar scaffolds.',
      icon: Sparkles
    },
    {
      id: 'pareto' as const,
      num: '05',
      title: 'Pareto Laboratory',
      subtitle: 'Multi-Objective Trade-Off Frontier',
      badge: 'Non-Dominated Sorting',
      description: 'Simultaneous 4-objective optimization balancing biological potency, aqueous solubility, toxicity risk, and drug-likeness.',
      details: 'Interactive 2D & 3D scatter plots identifying the non-dominated Pareto frontier where no property can improve without sacrificing another.',
      icon: GitCompare
    },
    {
      id: 'active_learning' as const,
      num: '06',
      title: 'Active Learning Queue',
      subtitle: 'Human-in-the-Loop Discovery',
      badge: 'BoTorch Gaussian Process',
      description: 'Iterative Bayesian optimization simulating compound evaluation rounds using acquisition functions.',
      details: 'Employs Upper Confidence Bound (UCB), Expected Improvement (EI), and Uncertainty Sampling to iteratively acquire high-value candidates.',
      icon: Cpu
    }
  ];

  return (
    <div className="relative min-h-screen bg-[#060709] text-slate-100 flex flex-col font-sans selection:bg-white selection:text-black overflow-hidden">
      
      {/* Editorial Floating Navigation Bar */}
      <header className="sticky top-0 z-40 bg-[#060709]/75 backdrop-blur-2xl border-b border-white/[0.07]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            
            {/* Wordmark */}
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-white/[0.05] border border-white/[0.12] flex items-center justify-center">
                <Atom className="w-4 h-4 text-white" />
              </div>
              <div className="flex flex-col">
                <span className="text-base font-medium tracking-tight text-white">
                  MoleculeMind
                </span>
                <span className="text-[10px] tracking-wider uppercase text-slate-400 font-mono">
                  Autonomous Molecular Intelligence
                </span>
              </div>
            </div>

            {/* Quick Links */}
            <div className="hidden md:flex items-center gap-8 text-xs font-medium text-slate-300">
              <a href="#philosophy" className="hover:text-white transition-colors">Philosophy</a>
              <a href="#showcase" className="hover:text-white transition-colors">Interactive Showcase</a>
              <a href="#modules" className="hover:text-white transition-colors">Platform Modules</a>
              <a href="#benchmarks" className="hover:text-white transition-colors">Benchmarks</a>
            </div>

            {/* CTA */}
            <div className="flex items-center gap-3">
              <button
                onClick={onOpenGlossary}
                className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium text-slate-300 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.1] transition-all cursor-pointer"
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Theory Guide</span>
              </button>

              <button
                onClick={() => handleLaunchWithConfetti('analyzer')}
                className="flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-semibold text-zinc-950 bg-white hover:bg-slate-200 transition-all shadow-lg cursor-pointer"
              >
                <span>Launch Workbench</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

          </div>
        </div>
      </header>

      {/* Atmospheric Diffused Background Lighting */}
      <div className="absolute top-10 left-1/2 -translate-x-1/2 w-[900px] h-[550px] ambient-glow-silver blur-[160px] pointer-events-none z-0" />
      <div className="absolute top-[600px] left-1/4 w-[600px] h-[400px] ambient-glow-cyan blur-[180px] pointer-events-none z-0" />
      <div className="absolute top-[1200px] right-1/4 w-[700px] h-[450px] ambient-glow-emerald blur-[180px] pointer-events-none z-0" />

      {/* Hero Section */}
      <section className="relative z-10 pt-16 sm:pt-24 pb-20 sm:pb-32 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          
          {/* Left Column: Editorial Headline & Narrative */}
          <div className="lg:col-span-7 space-y-8">
            
            {/* Kicker Pill */}
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.1] text-slate-300 text-[11px] font-mono tracking-wider uppercase"
            >
              <span className={`w-1.5 h-1.5 rounded-full ${backendHealthy ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              <span>Uncertainty-Aware Drug Discovery • v2.4</span>
            </motion.div>

            {/* Giant Editorial Headline */}
            <motion.div
              initial={{ opacity: 0, y: 25 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="space-y-2"
            >
              <h1 className="text-4xl sm:text-6xl lg:text-7xl font-medium tracking-tight text-white editorial-title">
                Molecular Design.
              </h1>
              <h1 className="text-4xl sm:text-6xl lg:text-7xl font-medium tracking-tight text-slate-400 editorial-title">
                Elevated by Certainty.
              </h1>
            </motion.div>

            {/* Narrative Paragraph */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="text-base sm:text-lg text-slate-300 max-w-2xl leading-relaxed font-normal"
            >
              Drug discovery is fraught with uncalibrated predictions and out-of-distribution hallucinations. 
              MoleculeMind integrates <strong className="text-white font-medium">Delaney ESOL regression</strong>, 
              <strong className="text-white font-medium"> finite-sample conformal prediction intervals</strong>, and 
              <strong className="text-white font-medium"> human-in-the-loop Bayesian active learning</strong> to transform computational hypotheses into confident laboratory decisions.
            </motion.p>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.3 }}
              className="flex flex-wrap items-center gap-4 pt-2"
            >
              <button
                onClick={() => handleLaunchWithConfetti('analyzer')}
                className="px-8 py-4 rounded-full bg-white text-zinc-950 font-semibold text-sm hover:bg-slate-200 transition-all shadow-[0_4px_25px_rgba(255,255,255,0.18)] hover:scale-[1.02] active:scale-[0.98] cursor-pointer flex items-center gap-2.5"
              >
                <span>Launch Interactive Workbench</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                onClick={onOpenGlossary}
                className="px-6 py-4 rounded-full bg-white/[0.05] hover:bg-white/[0.09] text-white border border-white/[0.12] hover:border-white/25 text-sm font-medium transition-all cursor-pointer flex items-center gap-2"
              >
                <BookOpen className="w-4 h-4 text-slate-300" />
                <span>Presentation Methodology</span>
              </button>
            </motion.div>

            {/* Quick Metrics Bar */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="grid grid-cols-3 gap-6 pt-6 border-t border-white/[0.08] max-w-lg"
            >
              <div>
                <div className="text-2xl sm:text-3xl font-light text-white font-mono">0.894</div>
                <div className="text-[11px] text-slate-400 font-mono tracking-wider uppercase mt-1">Test R² Score</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-light text-emerald-400 font-mono">95.0%</div>
                <div className="text-[11px] text-slate-400 font-mono tracking-wider uppercase mt-1">Conformal Coverage</div>
              </div>
              <div>
                <div className="text-2xl sm:text-3xl font-light text-cyan-400 font-mono">1,128</div>
                <div className="text-[11px] text-slate-400 font-mono tracking-wider uppercase mt-1">ESOL Compounds</div>
              </div>
            </motion.div>

          </div>

          {/* Right Column: Interactive 3D Molecular Sculpture */}
          <div className="lg:col-span-5 relative flex items-center justify-center">
            <div className="relative w-full aspect-square max-w-[500px]">
              
              {/* Subtle Ambient Background Halo */}
              <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-cyan-500/10 via-white/[0.03] to-emerald-500/10 blur-3xl" />
              
              {/* Three.js Interactive WebGL Orb */}
              <div className="relative w-full h-full z-10">
                <QuantumMolecularOrb className="w-full h-full" particleCount={1400} interactive={true} />
              </div>

              {/* Minimal Floating Indicator Pill */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-[#0c0d14]/80 border border-white/[0.1] backdrop-blur-xl text-[11px] text-slate-400 font-mono z-20 flex items-center gap-2 pointer-events-none">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span>Interactive 3D Molecular Conformer (Drag to Orbit)</span>
              </div>

            </div>
          </div>

        </div>
      </section>

      {/* Infinite Horizontal Scientific Marquee */}
      <section className="relative z-10 py-5 bg-[#0a0b10] border-y border-white/[0.06] overflow-hidden">
        <div className="animate-marquee flex items-center gap-12 text-xs font-mono text-slate-400 tracking-wider uppercase whitespace-nowrap">
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>Delaney ESOL Benchmark (N=1,128)</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Inductive Conformal Prediction P(Y ∈ [L, U]) ≥ 1 - α</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            <span>BoTorch SingleTaskGP Active Learning</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            <span>OECD Guidance Document No. 69 Applicability Domain</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-white" />
            <span>RDKit MMFF94 3D Energy Minimization</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>Pareto Non-Dominated Sorting Multi-Objective</span>
          </span>
          <span>•</span>
          {/* Duplicate for infinite seamless scroll */}
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>Delaney ESOL Benchmark (N=1,128)</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            <span>Inductive Conformal Prediction P(Y ∈ [L, U]) ≥ 1 - α</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            <span>BoTorch SingleTaskGP Active Learning</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            <span>OECD Guidance Document No. 69 Applicability Domain</span>
          </span>
          <span>•</span>
          <span className="flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-white" />
            <span>RDKit MMFF94 3D Energy Minimization</span>
          </span>
          <span>•</span>
        </div>
      </section>

      {/* Section 2: The Core Philosophy / 3 Structural Flaws */}
      <section id="philosophy" className="relative z-10 py-24 sm:py-32 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Heading */}
        <div className="max-w-3xl mb-16 space-y-4">
          <div className="editorial-subhead text-cyan-400">The Problem Paradigm</div>
          <h2 className="text-3xl sm:text-5xl font-medium tracking-tight text-white editorial-title">
            Why Conventional Molecular ML Fails in the Wet Lab.
          </h2>
          <p className="text-base text-slate-400 leading-relaxed">
            Standard drug discovery pipelines deploy black-box models that generate confident-looking point predictions. 
            When taken to synthesis, over 90% of candidates fail due to three fundamental flaws.
          </p>
        </div>

        {/* 3 Editorial Numbered Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          <div className="p-8 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] hover:border-white/20 transition-all duration-300 space-y-5">
            <div className="text-2xl font-light text-slate-400 font-mono">01 / 03</div>
            <h3 className="text-xl font-medium text-white tracking-tight">
              The Illusion of Point Estimates
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              A model predicting aqueous solubility at <code className="text-slate-300 font-mono">-2.5 logS</code> conceals whether the true laboratory value lies within ±0.3 or ±3.0. Without distribution-free prediction intervals, chemists cannot quantify experiment risk.
            </p>
            <div className="pt-2 text-xs font-mono text-cyan-400">
              Solution: Split Conformal Calibration
            </div>
          </div>

          <div className="p-8 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] hover:border-white/20 transition-all duration-300 space-y-5">
            <div className="text-2xl font-light text-slate-400 font-mono">02 / 03</div>
            <h3 className="text-xl font-medium text-white tracking-tight">
              Out-of-Distribution Hallucinations
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              When a novel molecular scaffold is presented, regression trees and neural networks silently extrapolate far beyond their training chemical manifold, producing statistically meaningless numbers with false confidence.
            </p>
            <div className="pt-2 text-xs font-mono text-amber-400">
              Solution: OECD Guidance No. 69 Leverage Cutoff
            </div>
          </div>

          <div className="p-8 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] hover:border-white/20 transition-all duration-300 space-y-5">
            <div className="text-2xl font-light text-slate-400 font-mono">03 / 03</div>
            <h3 className="text-xl font-medium text-white tracking-tight">
              Multi-Objective Conflict
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              Maximizing EGFR binding affinity frequently drives up lipophilicity (LogP), which ruins aqueous solubility and induces off-target toxicity. Real drug discovery requires non-dominated Pareto trade-off navigation.
            </p>
            <div className="pt-2 text-xs font-mono text-emerald-400">
              Solution: 4-Objective Pareto Frontier Sorting
            </div>
          </div>

        </div>
      </section>

      {/* Section 3: Interactive Compound Showcase */}
      <section id="showcase" className="relative z-10 py-20 bg-[#090a0f] border-y border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
            <div>
              <div className="editorial-subhead text-emerald-400">Live Interactive Demonstration</div>
              <h2 className="text-3xl sm:text-4xl font-medium tracking-tight text-white editorial-title mt-2">
                Evaluate Real Chemical Structures.
              </h2>
              <p className="text-sm text-slate-400 mt-2">
                Click any compound to inspect its real-time Delaney ESOL prediction, calibrated interval, and domain reliability.
              </p>
            </div>

            {/* Presets Selector Pills */}
            <div className="flex flex-wrap items-center gap-2">
              {PRESET_SHOWCASE.map((compound) => (
                <button
                  key={compound.name}
                  onClick={() => setSelectedDemo(compound)}
                  className={`px-4 py-2 rounded-full text-xs font-medium transition-all cursor-pointer ${
                    selectedDemo.name === compound.name
                      ? 'bg-white text-zinc-950 font-semibold shadow-md'
                      : 'bg-white/[0.04] text-slate-300 hover:bg-white/[0.08] border border-white/[0.08]'
                  }`}
                >
                  {compound.name}
                </button>
              ))}
            </div>
          </div>

          {/* Showcase Display Card */}
          <div className="p-8 sm:p-10 rounded-3xl bg-[#0e1017]/90 border border-white/[0.1] shadow-2xl backdrop-blur-2xl">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
              
              {/* Molecule Summary */}
              <div className="lg:col-span-6 space-y-6">
                <div className="flex items-center gap-3">
                  <h3 className="text-2xl sm:text-3xl font-medium text-white tracking-tight">
                    {selectedDemo.name}
                  </h3>
                  <span className="text-xs font-mono px-3 py-1 rounded-full bg-white/[0.06] border border-white/[0.1] text-slate-300">
                    {selectedDemo.formula}
                  </span>
                  <span className={`text-xs font-mono font-semibold px-3 py-1 rounded-full border ${selectedDemo.badgeClass}`}>
                    {selectedDemo.domain}
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-black/40 border border-white/[0.06] font-mono text-xs text-slate-300 break-all">
                  <span className="text-slate-400 mr-2">SMILES:</span>
                  {selectedDemo.smiles}
                </div>

                <p className="text-sm text-slate-300 leading-relaxed">
                  {selectedDemo.desc}
                </p>

                <div className="flex items-center gap-4 pt-2">
                  <button
                    onClick={() => {
                      onLaunchApp('prediction');
                    }}
                    className="px-6 py-3 rounded-full bg-white text-zinc-950 text-xs font-semibold hover:bg-slate-200 transition-all shadow-md cursor-pointer flex items-center gap-2"
                  >
                    <span>Analyze in Prediction Workbench</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Live Metric Cards */}
              <div className="lg:col-span-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                
                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.07] space-y-2">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                    Predicted Solubility
                  </div>
                  <div className="text-3xl font-light text-white font-mono">
                    {selectedDemo.logS} <span className="text-xs text-slate-400 font-sans">log(mol/L)</span>
                  </div>
                  <div className="text-xs text-slate-400">
                    Delaney ESOL ExtraTrees regression estimate
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.07] space-y-2">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider flex items-center justify-between">
                    <span>90% Conformal Interval</span>
                    <span className="text-emerald-400">Guaranteed</span>
                  </div>
                  <div className="text-2xl font-light text-emerald-300 font-mono">
                    {selectedDemo.conf90}
                  </div>
                  <div className="text-xs text-slate-400">
                    Nonconformity residual calibrated coverage
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.07] space-y-2">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                    Chemical Domain Status
                  </div>
                  <div className="text-lg font-medium text-white flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${selectedDemo.domain === 'IN_DOMAIN' ? 'bg-emerald-400' : selectedDemo.domain === 'BORDERLINE' ? 'bg-amber-400' : 'bg-rose-400'}`} />
                    <span>{selectedDemo.domain}</span>
                  </div>
                  <div className="text-xs text-slate-400">
                    OECD h* statistical leverage assessment
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/[0.07] space-y-2">
                  <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                    Inference Latency
                  </div>
                  <div className="text-2xl font-light text-cyan-300 font-mono">
                    &lt; 38 ms
                  </div>
                  <div className="text-xs text-slate-400">
                    Morgan 1024-bit + 8 physicochemical features
                  </div>
                </div>

              </div>

            </div>
          </div>

        </div>
      </section>

      {/* Section 4: The 6 Platform Discovery Modules */}
      <section id="modules" className="relative z-10 py-24 sm:py-32 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Section Heading */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-16">
          <div className="max-w-2xl space-y-4">
            <div className="editorial-subhead text-cyan-400">Interactive Architecture</div>
            <h2 className="text-3xl sm:text-5xl font-medium tracking-tight text-white editorial-title">
              Six Specialized Discovery Engines.
            </h2>
            <p className="text-base text-slate-400 leading-relaxed">
              Each module addresses a distinct scientific challenge, connected through a seamless data pipeline.
            </p>
          </div>

          <button
            onClick={() => handleLaunchWithConfetti('analyzer')}
            className="px-6 py-3 rounded-full bg-white/[0.06] hover:bg-white/[0.12] text-white border border-white/[0.12] text-xs font-semibold transition-all cursor-pointer flex items-center gap-2 self-start md:self-auto"
          >
            <span>Open All Modules</span>
            <ArrowUpRight className="w-4 h-4" />
          </button>
        </div>

        {/* 6 Capabilities Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {capabilities.map((module) => {
            const Icon = module.icon;
            return (
              <div
                key={module.id + module.num}
                className="group p-8 rounded-3xl bg-[#0c0d14]/70 border border-white/[0.08] hover:border-white/25 transition-all duration-300 flex flex-col justify-between"
              >
                <div className="space-y-5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-slate-400">
                      {module.num} / 06
                    </span>
                    <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-white/[0.05] border border-white/[0.08] text-slate-300">
                      {module.badge}
                    </span>
                  </div>

                  <div className="w-10 h-10 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-white group-hover:bg-white group-hover:text-black transition-all duration-300">
                    <Icon className="w-5 h-5" />
                  </div>

                  <div className="space-y-1">
                    <h3 className="text-lg font-medium text-white tracking-tight">
                      {module.title}
                    </h3>
                    <div className="text-xs text-slate-400 font-mono">
                      {module.subtitle}
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">
                    {module.description}
                  </p>

                  <p className="text-[11px] text-slate-400 leading-relaxed pt-2 border-t border-white/[0.05]">
                    {module.details}
                  </p>
                </div>

                <div className="pt-6">
                  <button
                    onClick={() => handleLaunchWithConfetti(module.id)}
                    className="w-full py-2.5 rounded-xl bg-white/[0.03] hover:bg-white group-hover:text-zinc-950 text-slate-300 border border-white/[0.08] text-xs font-medium transition-all flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <span>Launch Module</span>
                    <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>

      </section>

      {/* Section 5: Empirical Benchmarks */}
      <section id="benchmarks" className="relative z-10 py-20 bg-[#08090e] border-t border-white/[0.06]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="max-w-2xl mb-12 space-y-4">
            <div className="editorial-subhead text-cyan-400">Scientific Validation</div>
            <h2 className="text-3xl sm:text-4xl font-medium tracking-tight text-white editorial-title">
              Trained on Rigorous Experimental Ground Truth.
            </h2>
            <p className="text-sm text-slate-400">
              Unlike theoretical toys, MoleculeMind is backed by peer-reviewed Delaney solubility datasets and calibrated against independent test distributions.
            </p>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] space-y-2">
              <div className="text-3xl sm:text-4xl font-light text-white font-mono">0.894</div>
              <div className="text-xs font-semibold text-slate-300">Test Set R² Metric</div>
              <div className="text-[11px] text-slate-400">ExtraTrees regression coefficient of determination</div>
            </div>

            <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] space-y-2">
              <div className="text-3xl sm:text-4xl font-light text-emerald-400 font-mono">0.704</div>
              <div className="text-xs font-semibold text-slate-300">Root Mean Square Error</div>
              <div className="text-[11px] text-slate-400">log(mol/L) units across held-out test split</div>
            </div>

            <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] space-y-2">
              <div className="text-3xl sm:text-4xl font-light text-cyan-400 font-mono">1,032</div>
              <div className="text-xs font-semibold text-slate-300">Total Input Features</div>
              <div className="text-[11px] text-slate-400">1,024 Morgan ECFP4 bits + 8 RDKit descriptors</div>
            </div>

            <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.08] space-y-2">
              <div className="text-3xl sm:text-4xl font-light text-purple-400 font-mono">BoTorch</div>
              <div className="text-xs font-semibold text-slate-300">Gaussian Process Surrogate</div>
              <div className="text-[11px] text-slate-400">SingleTaskGP active learning acquisition engine</div>
            </div>
          </div>

          <div className="mt-12 p-6 rounded-2xl bg-[#0c0e15] border border-white/[0.08] flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Sparkles className="w-5 h-5 text-cyan-400 shrink-0" />
              <div className="text-xs text-slate-300">
                Want to inspect the complete Delaney ESOL parity curve, residual distributions, and test predictions?
              </div>
            </div>
            <button
              onClick={() => handleLaunchWithConfetti('benchmark')}
              className="px-5 py-2.5 rounded-full bg-white text-zinc-950 text-xs font-semibold hover:bg-slate-200 transition-all shrink-0 cursor-pointer flex items-center gap-2"
            >
              <span>View ESOL Benchmark Tab</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

        </div>
      </section>

      {/* Footer / Final Launch CTA */}
      <footer className="relative z-10 py-16 bg-[#040507] border-t border-white/[0.08] text-slate-400 text-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Atom className="w-4 h-4 text-white" />
            <span className="text-slate-200 font-medium font-mono">MoleculeMind Decision Platform</span>
            <span>—</span>
            <span className="text-slate-400">RDKit · PyTorch · BoTorch · Scikit-learn · 3Dmol.js</span>
          </div>

          <div className="flex items-center gap-6 text-slate-400">
            <button onClick={onOpenGlossary} className="hover:text-white transition-colors cursor-pointer">
              Scientific Glossary
            </button>
            <button onClick={() => handleLaunchWithConfetti('analyzer')} className="hover:text-white transition-colors cursor-pointer text-white font-medium">
              Open Workbench →
            </button>
          </div>
        </div>
      </footer>

    </div>
  );
};
