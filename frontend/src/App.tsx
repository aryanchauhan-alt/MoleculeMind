import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LandingPage } from './components/LandingPage';
import { PropertyPredictionTab } from './components/PropertyPredictionTab';
import { ParetoLaboratoryTab } from './components/ParetoLaboratoryTab';
import { ActiveLearningTab } from './components/ActiveLearningTab';
import { BenchmarkTab } from './components/BenchmarkTab';
import { DiscoveryEngineTab } from './components/DiscoveryEngineTab';
import { ConceptHelpModal } from './components/ScientificTooltip';
import { AgencyCursor } from './components/AgencyCursor';
import { SceneProvider, useScene } from './components/instrument/SceneCanvas';
import { Nav } from './components/instrument/Nav';
import { AnalyzerModule } from './components/instrument/AnalyzerModule';
import { BootSequence } from './components/instrument/primitives';
import { Atom } from 'lucide-react';

function AppInner() {
  const scene = useScene();
  const [currentView, setCurrentView] = useState<'landing' | 'app'>('landing');
  const [activeTab, setActiveTab] = useState<'analyzer' | 'prediction' | 'pareto' | 'discovery' | 'active_learning' | 'benchmark'>('analyzer');
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [activeSmiles, setActiveSmiles] = useState<string>('CC(=O)Oc1ccccc1C(=O)O');
  const [isGlossaryOpen, setIsGlossaryOpen] = useState<boolean>(false);
  const [booted, setBooted] = useState<boolean>(false);

  // URL Hash Synchronisation for bookmarking and back button navigation
  useEffect(() => {
    const parseHash = () => {
      const hash = window.location.hash;
      if (hash.startsWith('#app')) {
        setCurrentView('app');
        if (hash.includes('tab=')) {
          const tab = hash.split('tab=')[1]?.split('&')[0];
          if (['analyzer', 'prediction', 'pareto', 'discovery', 'active_learning', 'benchmark'].includes(tab)) {
            setActiveTab(tab as any);
          }
        }
      } else {
        setCurrentView('landing');
      }
    };

    parseHash();
    window.addEventListener('popstate', parseHash);
    return () => window.removeEventListener('popstate', parseHash);
  }, []);

  const updateHash = (view: 'landing' | 'app', tab?: string) => {
    if (view === 'landing') {
      window.history.pushState(null, '', '#');
    } else {
      window.history.pushState(null, '', `#app?tab=${tab || activeTab}`);
    }
  };

  const handleLaunchApp = (tab?: 'analyzer' | 'prediction' | 'pareto' | 'discovery' | 'active_learning' | 'benchmark') => {
    const targetTab = tab || 'analyzer';
    setActiveTab(targetTab);
    setCurrentView('app');
    updateHash('app', targetTab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleReturnToLanding = () => {
    setCurrentView('landing');
    updateHash('landing');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleTabChange = (tab: 'analyzer' | 'prediction' | 'pareto' | 'discovery' | 'active_learning' | 'benchmark') => {
    setActiveTab(tab);
    scene.setMode(tab);
    updateHash('app', tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        setBackendHealthy(true);
      } else {
        setBackendHealthy(false);
      }
    } catch {
      setBackendHealthy(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendToPredict = (smiles: string) => {
    setActiveSmiles(smiles);
    handleTabChange('prediction');
  };

  const handleSelectCandidate = (smiles: string) => {
    setActiveSmiles(smiles);
    handleTabChange('analyzer');
  };

  return (
    <div className="min-h-screen flex flex-col in-grain relative" style={{ color: 'var(--in-text)' }}>
      {/* Editorial Luxury Agency Cursor */}
      <AgencyCursor />

      {/* Boot sequence — covers the first paint while models load */}
      <AnimatePresence>
        {!booted && <BootSequence ready={backendHealthy} onDone={() => setBooted(true)} />}
      </AnimatePresence>

      {/* Measurement grid, very faint, behind the WebGL layer */}
      <div className="fixed inset-0 in-grid-bg pointer-events-none z-0 opacity-50" />

      {/* View 1: Editorial Landing Page */}
      {currentView === 'landing' ? (
        <LandingPage
          onLaunchApp={handleLaunchApp}
          onOpenGlossary={() => setIsGlossaryOpen(true)}
          backendHealthy={backendHealthy}
        />
      ) : (
        /* View 2: High-Fidelity Scientific Discovery Workbench */
        <div className="min-h-screen flex flex-col relative z-10">
          {/* Module index */}
          <Nav
            active={activeTab as any}
            onChange={(id) => handleTabChange(id as any)}
            backendHealthy={backendHealthy}
            onHome={handleReturnToLanding}
          />

          {/* Main Discovery Container */}
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">

            {/* Animated Tab Views */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
              >
                {activeTab === 'analyzer' && (
                  <AnalyzerModule onSendToPrediction={handleSendToPredict} />
                )}

                {activeTab === 'prediction' && (
                  <PropertyPredictionTab 
                    initialSmiles={activeSmiles} 
                    onSendToPareto={() => handleTabChange('pareto')}
                  />
                )}

                {activeTab === 'pareto' && (
                  <ParetoLaboratoryTab 
                    onSelectCandidate={handleSelectCandidate} 
                  />
                )}

                {activeTab === 'discovery' && <DiscoveryEngineTab />}

                {activeTab === 'active_learning' && (
                  <ActiveLearningTab 
                    onSelectCandidate={handleSelectCandidate} 
                  />
                )}

                {activeTab === 'benchmark' && (
                  <BenchmarkTab />
                )}
              </motion.div>
            </AnimatePresence>
          </main>

          {/* Scientific Platform Editorial Footer */}
          <footer className="mt-auto border-t border-white/[0.06] bg-[#050608]/90 backdrop-blur-xl py-6 text-xs text-slate-400 font-mono">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-center sm:text-left">
              <div className="flex items-center gap-2.5">
                <Atom className="w-4 h-4 text-white" />
                <span className="text-slate-200 font-medium">MoleculeMind Decision Platform</span>
                <span className="text-slate-600">//</span>
                <span className="text-slate-400">RDKit · PyTorch · BoTorch · Scikit-learn · 3Dmol.js</span>
              </div>

              <div className="flex items-center gap-4 text-[11px] text-slate-400">
                <button 
                  onClick={handleReturnToLanding} 
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Return to Overview
                </button>
                <span>·</span>
                <button 
                  onClick={() => setIsGlossaryOpen(true)} 
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Scientific Methodology
                </button>
                <span>·</span>
                <span className="text-slate-300 font-medium">P(Y ∈ [L, U]) ≥ 1 - α</span>
              </div>
            </div>
          </footer>
        </div>
      )}

      {/* Global Scientific Glossary & Theory Modal */}
      <ConceptHelpModal 
        isOpen={isGlossaryOpen} 
        onClose={() => setIsGlossaryOpen(false)} 
      />
    </div>
  );
}

export function App() {
  return (
    <SceneProvider>
      <AppInner />
    </SceneProvider>
  );
}

export default App;
