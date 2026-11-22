import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LandingPage } from './components/LandingPage';
import { ConceptHelpModal } from './components/ScientificTooltip';
import { AgencyCursor } from './components/AgencyCursor';
import { SceneProvider, useScene } from './components/instrument/SceneCanvas';
import { Nav } from './components/instrument/Nav';
import { AnalyzerModule } from './components/instrument/AnalyzerModule';
import { PredictionModule } from './components/instrument/PredictionModule';
import { ParetoModule } from './components/instrument/ParetoModule';
import { DiscoveryModule } from './components/instrument/DiscoveryModule';
import { CampaignModule } from './components/instrument/CampaignModule';
import { BenchmarkModule } from './components/instrument/BenchmarkModule';
import { BootSequence } from './components/instrument/primitives';

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
    // `hashchange` matters as well as `popstate`: navigating to a different
    // hash on the same document fires only the former, so without it a pasted
    // or bookmarked module link left the previous view on screen.
    window.addEventListener('popstate', parseHash);
    window.addEventListener('hashchange', parseHash);
    return () => {
      window.removeEventListener('popstate', parseHash);
      window.removeEventListener('hashchange', parseHash);
    };
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
    <div className="min-h-screen flex flex-col relative" style={{ background: 'var(--pp-paper)', color: 'var(--pp-ink)' }}>
      {/* Editorial Luxury Agency Cursor */}
      <AgencyCursor />

      {/* Boot sequence — covers the first paint while models load */}
      <AnimatePresence>
        {!booted && <BootSequence ready={backendHealthy} onDone={() => setBooted(true)} />}
      </AnimatePresence>


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
                  <PredictionModule initialSmiles={activeSmiles} />
                )}

                {activeTab === 'pareto' && (
                  <ParetoModule onSelect={handleSelectCandidate} />
                )}

                {activeTab === 'discovery' && <DiscoveryModule />}

                {activeTab === 'active_learning' && (
                  <CampaignModule onSelect={handleSelectCandidate} />
                )}

                {activeTab === 'benchmark' && <BenchmarkModule />}
              </motion.div>
            </AnimatePresence>
          </main>

          {/* Scientific Platform Editorial Footer */}
          <footer
            className="mt-auto"
            style={{ borderTop: '1px solid var(--pp-rule)', background: 'var(--pp-paper)' }}
          >
            <div className="max-w-[1400px] mx-auto px-5 sm:px-8 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">
              <div>
                <div className="in-display text-sm">MoleculeMind</div>
                <div className="in-label mt-1.5">
                  RDKit · scikit-learn · PyTorch · BoTorch · FastAPI
                </div>
              </div>
              <div className="flex items-center gap-5">
                <button
                  onClick={handleReturnToLanding}
                  className="in-label hover:opacity-60 transition-opacity cursor-pointer"
                >
                  overview
                </button>
                <button
                  onClick={() => setIsGlossaryOpen(true)}
                  className="in-label hover:opacity-60 transition-opacity cursor-pointer"
                >
                  methodology
                </button>
                <span className="in-num text-[10px]" style={{ color: 'var(--pp-faint)' }}>
                  P(Y &isin; [L, U]) &ge; 1 &minus; &alpha;
                </span>
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
