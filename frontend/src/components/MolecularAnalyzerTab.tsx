import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MoleculeViewer3D } from './MoleculeViewer3D';
import { ScientificTooltip } from './ScientificTooltip';
import { 
  FlaskConical, 
  Search, 
  AlertTriangle, 
  ShieldCheck, 
  Sparkles, 
  Layers, 
  Box, 
  ArrowRight,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface AnalyzerProps {
  onAnalyzeComplete?: (data: any) => void;
  onSendToPredict?: (smiles: string) => void;
}

const SAMPLE_MOLECULES = [
  { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O', category: 'Analgesic' },
  { name: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C', category: 'Stimulant' },
  { name: 'Paracetamol', smiles: 'CC(=O)Nc1ccc(O)cc1', category: 'Antipyretic' },
  { name: 'Ibuprofen', smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', category: 'NSAID' },
  { name: 'Nicotinamide', smiles: 'NC(=O)c1cccnc1', category: 'Vitamin B3' },
  { name: 'Diclofenac', smiles: 'O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl', category: 'NSAID' },
  { name: 'Imatinib core', smiles: 'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1', category: 'Kinase Inhibitor' }
];

export const MolecularAnalyzerTab: React.FC<AnalyzerProps> = ({ onSendToPredict }) => {
  const [smilesInput, setSmilesInput] = useState<string>('CC(=O)Oc1ccccc1C(=O)O');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);
  const [viewMode, setViewMode] = useState<'both' | '2d' | '3d'>('both');
  const [showAdvancedDescriptors, setShowAdvancedDescriptors] = useState<boolean>(false);

  const handleAnalyze = async (smilesToAnalyze?: string) => {
    const targetSmiles = smilesToAnalyze || smilesInput;
    if (!targetSmiles.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: targetSmiles.trim() })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze molecule');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Error occurred while contacting backend');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleAnalyze('CC(=O)Oc1ccccc1C(=O)O');
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      
      {/* Top Input & Preset Selection Card */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-mono tracking-wider uppercase text-slate-400">
              <FlaskConical className="w-3.5 h-3.5 text-cyan-400" />
              <span>Stage 01 • Molecular Verification Engine</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-medium tracking-tight text-white">
              Chemical Structure & 3D Conformation Analysis
            </h2>
          </div>

          {result && onSendToPredict && (
            <button
              onClick={() => onSendToPredict(result.smiles)}
              className="px-5 py-2.5 rounded-full bg-white text-zinc-950 font-semibold text-xs hover:bg-slate-200 transition-all shadow-md shrink-0 cursor-pointer flex items-center gap-2 self-start md:self-auto"
            >
              <span>Predict Properties</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* SMILES Input Form */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <input
              type="text"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
              placeholder="Enter canonical or isomeric SMILES (e.g. CC(=O)Oc1ccccc1C(=O)O)..."
              className="w-full bg-black/40 border border-white/[0.1] rounded-2xl px-5 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-white/30 focus:ring-1 focus:ring-white/20 font-mono text-sm transition-all"
            />
          </div>

          <button
            onClick={() => handleAnalyze()}
            disabled={loading}
            className="px-7 py-3.5 rounded-2xl bg-white hover:bg-slate-200 disabled:opacity-50 text-zinc-950 font-semibold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer shadow-lg shrink-0"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-zinc-950 border-t-transparent rounded-full animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            <span>Analyze Molecule</span>
          </button>
        </div>

        {/* Preset Molecule Pills */}
        <div className="flex flex-wrap items-center gap-2 mt-5 pt-4 border-t border-white/[0.06]">
          <span className="text-xs font-mono text-slate-400 flex items-center gap-1.5 mr-1">
            <Sparkles className="w-3.5 h-3.5 text-slate-300" /> Presets:
          </span>
          {SAMPLE_MOLECULES.map((mol) => (
            <button
              key={mol.name}
              onClick={() => {
                setSmilesInput(mol.smiles);
                handleAnalyze(mol.smiles);
              }}
              className="px-3.5 py-1.5 text-xs rounded-full bg-white/[0.03] hover:bg-white/[0.08] text-slate-300 hover:text-white transition-all border border-white/[0.08] hover:border-white/20 cursor-pointer flex items-center gap-1.5"
            >
              <span>{mol.name}</span>
              <span className="text-[10px] text-slate-400 font-mono">({mol.category})</span>
            </button>
          ))}
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mt-5 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs sm:text-sm flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
            <div>
              <strong>Validation Error:</strong> {error}
            </div>
          </div>
        )}
      </div>

      {/* Main Analysis Display */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Molecular Structure Visualizers (2D & 3D) */}
          <div className="lg:col-span-7 space-y-8">
            <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6 pb-4 border-b border-white/[0.06]">
                <div className="flex items-center gap-2.5">
                  <Layers className="w-5 h-5 text-slate-300" />
                  <div>
                    <h3 className="font-medium text-white text-base">Molecular Topologies</h3>
                    <p className="text-xs text-slate-400">RDKit 2D vector depiction & MMFF94 force-field 3D conformer</p>
                  </div>
                </div>

                {/* View Mode Switcher */}
                <div className="flex bg-black/40 border border-white/[0.08] rounded-full p-1 text-xs self-start sm:self-auto">
                  <button
                    onClick={() => setViewMode('both')}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-all ${
                      viewMode === 'both' ? 'bg-white text-zinc-950 font-semibold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Split View
                  </button>
                  <button
                    onClick={() => setViewMode('2d')}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-all ${
                      viewMode === '2d' ? 'bg-white text-zinc-950 font-semibold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    2D Only
                  </button>
                  <button
                    onClick={() => setViewMode('3d')}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-all ${
                      viewMode === '3d' ? 'bg-white text-zinc-950 font-semibold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    3D Conformer
                  </button>
                </div>
              </div>

              {/* Viewports Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 2D Structure Display */}
                {(viewMode === 'both' || viewMode === '2d') && (
                  <div className={`rounded-2xl bg-black/40 border border-white/[0.06] p-4 flex flex-col items-center justify-center relative min-h-[300px] ${
                    viewMode === '2d' ? 'md:col-span-2 min-h-[360px]' : ''
                  }`}>
                    <div className="absolute top-3 left-3 text-[10px] font-mono text-slate-400 tracking-wider uppercase bg-white/[0.04] px-2.5 py-1 rounded-full border border-white/[0.08]">
                      2D Vector SVG
                    </div>
                    <div 
                      className="w-full flex items-center justify-center pt-4"
                      dangerouslySetInnerHTML={{ __html: result.svg_2d }}
                    />
                  </div>
                )}

                {/* 3D Mol Viewer */}
                {(viewMode === 'both' || viewMode === '3d') && (
                  <div className={`flex flex-col min-h-[300px] ${viewMode === '3d' ? 'md:col-span-2 min-h-[360px]' : ''}`}>
                    <MoleculeViewer3D
                      sdfData={result.sdf_3d}
                      sdfError={result.sdf_error}
                      moleculeName={result.descriptors.formula}
                    />
                  </div>
                )}
              </div>

              {/* Canonical SMILES readout */}
              <div className="mt-5 p-3.5 bg-black/40 rounded-2xl border border-white/[0.06] flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
                <span className="text-slate-400 font-mono shrink-0">Validated Canonical SMILES:</span>
                <code className="text-slate-200 font-mono text-xs overflow-x-auto select-all bg-white/[0.04] px-3 py-1 rounded-lg border border-white/[0.05]">
                  {result.smiles}
                </code>
              </div>
            </div>

            {/* Drug-Likeness Assessment (Lipinski & Veber) */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              {/* Lipinski Rule of 5 */}
              <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 shadow-xl backdrop-blur-2xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <h4 className="font-medium text-white text-sm">Lipinski's Rule of 5</h4>
                    </div>
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold flex items-center gap-1 border ${
                      result.descriptors.lipinski.passed 
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                        : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                    }`}>
                      {result.descriptors.lipinski.passed ? (
                        <>
                          <CheckCircle className="w-3 h-3" />
                          <span>PASSED</span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-3 h-3" />
                          <span>VIOLATION</span>
                        </>
                      )}
                    </span>
                  </div>

                  <p className="text-xs text-slate-400 mb-4">
                    Evaluates oral bioavailability based on molecular weight, lipophilicity, and hydrogen bonding.
                  </p>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-white/[0.05]">
                      <span className="text-slate-400">Molecular Weight (≤ 500 Da)</span>
                      <span className={`font-mono font-medium ${result.descriptors.molecular_weight <= 500 ? 'text-white' : 'text-rose-400'}`}>
                        {result.descriptors.molecular_weight} Da
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-white/[0.05]">
                      <span className="text-slate-400">LogP Lipophilicity (≤ 5.0)</span>
                      <span className={`font-mono font-medium ${result.descriptors.logp <= 5.0 ? 'text-white' : 'text-rose-400'}`}>
                        {result.descriptors.logp}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-white/[0.05]">
                      <span className="text-slate-400">H-Bond Donors (≤ 5)</span>
                      <span className={`font-mono font-medium ${result.descriptors.hbd <= 5 ? 'text-white' : 'text-rose-400'}`}>
                        {result.descriptors.hbd}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5">
                      <span className="text-slate-400">H-Bond Acceptors (≤ 10)</span>
                      <span className={`font-mono font-medium ${result.descriptors.hba <= 10 ? 'text-white' : 'text-rose-400'}`}>
                        {result.descriptors.hba}
                      </span>
                    </div>
                  </div>
                </div>

                {result.descriptors.lipinski.violations.length > 0 ? (
                  <div className="mt-4 p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 font-mono">
                    <strong>Violations:</strong> {result.descriptors.lipinski.violations.join(', ')}
                  </div>
                ) : (
                  <div className="mt-4 text-xs text-emerald-400 flex items-center gap-1.5 font-mono">
                    <CheckCircle className="w-3.5 h-3.5" />
                    <span>0 Violations (Favorable Oral Absorption)</span>
                  </div>
                )}
              </div>

              {/* Veber Bioavailability Rule */}
              <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 shadow-xl backdrop-blur-2xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-cyan-400" />
                      <h4 className="font-medium text-white text-sm">Veber Criteria</h4>
                    </div>
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${
                      result.descriptors.veber.passed 
                        ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
                        : 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                    }`}>
                      {result.descriptors.veber.passed ? 'OPTIMAL' : 'BORDERLINE'}
                    </span>
                  </div>

                  <p className="text-xs text-slate-400 mb-4">
                    Assesses intestinal membrane permeability through rotatable bond flexibility and polar surface area.
                  </p>

                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-white/[0.05]">
                      <span className="text-slate-400">Rotatable Bonds (≤ 10)</span>
                      <span className={`font-mono font-medium ${result.descriptors.rotatable_bonds <= 10 ? 'text-white' : 'text-amber-400'}`}>
                        {result.descriptors.rotatable_bonds}
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-white/[0.05]">
                      <span className="text-slate-400">Polar Surface Area (≤ 140 Å²)</span>
                      <span className={`font-mono font-medium ${result.descriptors.tpsa <= 140 ? 'text-white' : 'text-amber-400'}`}>
                        {result.descriptors.tpsa} Å²
                      </span>
                    </div>
                    <div className="flex justify-between py-1.5">
                      <span className="text-slate-400">Oral Absorption Probability</span>
                      <span className="font-mono font-medium text-emerald-400">
                        {result.descriptors.veber.passed ? 'High Probability' : 'Moderate'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 text-xs text-slate-400 font-mono">
                  Veber criteria indicate favorable passive epithelial transit.
                </div>
              </div>

            </div>
          </div>

          {/* Right Column: Basic Information & Advanced Physicochemical Descriptors */}
          <div className="lg:col-span-5 space-y-8">
            <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/[0.06]">
                <div className="flex items-center gap-2.5">
                  <Box className="w-5 h-5 text-slate-300" />
                  <h3 className="font-medium text-white text-base">Compound Blueprint</h3>
                </div>
                <span className="text-xs font-mono text-slate-300 font-semibold px-2.5 py-1 rounded-full bg-white/[0.05] border border-white/[0.08]">
                  {result.descriptors.formula}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs mb-6">
                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono">Formula</div>
                  <div className="text-base font-light font-mono text-white mt-1">
                    {result.descriptors.formula}
                  </div>
                </div>

                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono">Heavy Atoms</div>
                  <div className="text-base font-light font-mono text-cyan-300 mt-1">
                    {result.descriptors.heavy_atom_count}
                  </div>
                </div>

                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono">Formal Charge</div>
                  <div className="text-base font-light font-mono text-white mt-1">
                    {result.descriptors.formal_charge}
                  </div>
                </div>

                <div className="bg-black/40 p-4 rounded-2xl border border-white/[0.06]">
                  <div className="text-slate-400 font-mono">
                    <ScientificTooltip id="qed">
                      <span className="underline decoration-dotted cursor-help">QED Score</span>
                    </ScientificTooltip>
                  </div>
                  <div className="text-base font-light font-mono text-emerald-400 mt-1">
                    {result.descriptors.qed}
                  </div>
                </div>
              </div>

              {/* Primary Physicochemical Descriptors */}
              <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Key Physicochemical Features
              </h4>

              <div className="space-y-2 text-xs">
                {/* Molecular Weight */}
                <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06] flex items-center justify-between">
                  <div>
                    <span className="text-white font-medium">Molecular Weight (MW)</span>
                    <p className="text-[11px] text-slate-400 font-mono">Standard dalton mass</p>
                  </div>
                  <span className="text-sm font-mono text-white">
                    {result.descriptors.molecular_weight} <span className="text-[10px] text-slate-400">g/mol</span>
                  </span>
                </div>

                {/* LogP */}
                <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06] flex items-center justify-between">
                  <div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-white font-medium">MolLogP (Octanol/Water)</span>
                      <ScientificTooltip id="solubility" />
                    </div>
                    <p className="text-[11px] text-slate-400 font-mono">Lipophilicity index</p>
                  </div>
                  <span className="text-sm font-mono text-cyan-300">
                    {result.descriptors.logp}
                  </span>
                </div>

                {/* TPSA */}
                <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06] flex items-center justify-between">
                  <div>
                    <span className="text-white font-medium">Polar Surface Area (TPSA)</span>
                    <p className="text-[11px] text-slate-400 font-mono">Cell membrane transit proxy</p>
                  </div>
                  <span className="text-sm font-mono text-white">
                    {result.descriptors.tpsa} <span className="text-[10px] text-slate-400">Å²</span>
                  </span>
                </div>
              </div>

              {/* Progressive Disclosure: Advanced Descriptors */}
              <div className="mt-5 pt-4 border-t border-white/[0.06]">
                <button
                  onClick={() => setShowAdvancedDescriptors(!showAdvancedDescriptors)}
                  className="w-full py-2.5 px-4 rounded-xl bg-black/40 hover:bg-white/[0.04] text-slate-300 hover:text-white text-xs font-mono flex items-center justify-between transition-colors border border-white/[0.06] cursor-pointer"
                >
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-slate-300" />
                    <span>{showAdvancedDescriptors ? 'Collapse Detailed Descriptors' : 'Expand Advanced Descriptors (Rings, Csp3, Rotatable)'}</span>
                  </span>
                  {showAdvancedDescriptors ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {showAdvancedDescriptors && (
                  <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                    <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06]">
                      <span className="text-slate-400 font-mono">HBD / HBA</span>
                      <div className="text-sm font-mono text-white mt-1">
                        {result.descriptors.hbd} / {result.descriptors.hba}
                      </div>
                    </div>

                    <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06]">
                      <span className="text-slate-400 font-mono">Rotatable Bonds</span>
                      <div className="text-sm font-mono text-white mt-1">
                        {result.descriptors.rotatable_bonds}
                      </div>
                    </div>

                    <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06]">
                      <span className="text-slate-400 font-mono">Ring Count / Aromatic</span>
                      <div className="text-sm font-mono text-white mt-1">
                        {result.descriptors.ring_count} / {result.descriptors.aromatic_ring_count}
                      </div>
                    </div>

                    <div className="p-3 bg-black/40 rounded-2xl border border-white/[0.06]">
                      <span className="text-slate-400 font-mono">Fraction Csp3</span>
                      <div className="text-sm font-mono text-white mt-1">
                        {result.descriptors.fraction_csp3}
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Forward to Predict CTA */}
              {onSendToPredict && (
                <div className="mt-6 p-5 rounded-2xl bg-white/[0.02] border border-white/[0.08] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <div className="text-xs font-medium text-white flex items-center gap-2">
                      <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                      Forward Molecule to Stage 02
                    </div>
                    <p className="text-[11px] text-slate-400 mt-1">
                      Compute Delaney ESOL solubility, conformal intervals, and OECD domain reliability.
                    </p>
                  </div>
                  <button
                    onClick={() => onSendToPredict(result.smiles)}
                    className="px-5 py-2.5 bg-white hover:bg-slate-200 text-zinc-950 font-semibold text-xs rounded-full shadow-md transition-all shrink-0 cursor-pointer flex items-center gap-2"
                  >
                    <span>Proceed</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

            </div>
          </div>

        </div>
      )}

    </motion.div>
  );
};
