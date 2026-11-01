import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Plot from 'react-plotly.js';
import { ScientificTooltip } from './ScientificTooltip';
import { 
  GitCompare, 
  Sparkles, 
  Award, 
  Filter, 
  ArrowUpRight,
  Info,
  ChevronDown,
  ChevronUp,
  TrendingUp,
  Search
} from 'lucide-react';

interface ParetoLabProps {
  onSelectCandidate?: (smiles: string) => void;
}

const PROPERTY_OPTIONS = [
  { key: 'bioactivity_pic50', label: 'Bioactivity (pIC50)', unit: 'potency', maximize: true },
  { key: 'solubility_logs', label: 'Aqueous Solubility (logS)', unit: 'log(mol/L)', maximize: true },
  { key: 'toxicity_risk_probability', label: 'Toxicity Risk Probability', unit: 'prob', maximize: false },
  { key: 'qed_drug_likeness', label: 'QED Drug-Likeness', unit: 'score', maximize: true },
  { key: 'metabolic_stability_half_life_min', label: 'Metabolic Half-Life (t1/2)', unit: 'min', maximize: true },
  { key: 'molecular_weight', label: 'Molecular Weight', unit: 'g/mol', maximize: false }
];

export const ParetoLaboratoryTab: React.FC<ParetoLabProps> = ({ onSelectCandidate }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<any | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<any | null>(null);
  const [xAxisKey, setXAxisKey] = useState<string>('bioactivity_pic50');
  const [yAxisKey, setYAxisKey] = useState<string>('solubility_logs');
  const [plotMode, setPlotMode] = useState<'2d' | '3d'>('2d');
  const [filterParetoOnly, setFilterParetoOnly] = useState<boolean>(false);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [showMathDetails, setShowMathDetails] = useState<boolean>(false);

  const fetchParetoData = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/pareto');
      const json = await res.json();
      setData(json);
      if (json.candidates && json.candidates.length > 0) {
        setSelectedCandidate(json.candidates[0]);
      }
    } catch (e) {
      console.error("Failed to load Pareto data:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchParetoData();
  }, []);

  if (loading) {
    return (
      <div className="p-12 text-center flex flex-col items-center justify-center space-y-4">
        <div className="w-8 h-8 border-2 border-white border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-400 font-mono text-xs">Computing non-dominated Pareto sorting over library candidates...</p>
      </div>
    );
  }

  if (!data || !data.candidates) {
    return (
      <div className="p-12 text-center bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl text-rose-300">
        Failed to calculate Pareto frontier. Verify backend connection.
      </div>
    );
  }

  const allCandidates = data.candidates || [];
  const xProp = PROPERTY_OPTIONS.find(p => p.key === xAxisKey) || PROPERTY_OPTIONS[0];
  const yProp = PROPERTY_OPTIONS.find(p => p.key === yAxisKey) || PROPERTY_OPTIONS[1];

  const filteredBySearch = allCandidates.filter((c: any) => 
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    c.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.smiles.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const displayedCandidates = filterParetoOnly 
    ? filteredBySearch.filter((c: any) => c.is_pareto_optimal)
    : filteredBySearch;

  const paretoPoints = allCandidates.filter((c: any) => c.is_pareto_optimal);
  const dominatedPoints = allCandidates.filter((c: any) => !c.is_pareto_optimal);

  const sortedPareto = [...paretoPoints].sort((a: any, b: any) => a[xAxisKey] - b[xAxisKey]);

  // 2D Plotly Traces (Obsidian Agency Palette)
  const traceDominated = {
    x: dominatedPoints.map((c: any) => c[xAxisKey]),
    y: dominatedPoints.map((c: any) => c[yAxisKey]),
    customdata: dominatedPoints.map((c: any) => c.id),
    text: dominatedPoints.map((c: any) => 
      `<b>${c.name}</b> (Dominated)<br>ID: ${c.id}<br>${xProp.label}: ${c[xAxisKey]}<br>${yProp.label}: ${c[yAxisKey]}<br>QED: ${c.qed_drug_likeness}<br>Pareto Rank: ${c.pareto_rank}`
    ),
    hoverinfo: 'text',
    mode: 'markers',
    type: 'scatter',
    name: 'Dominated Pool',
    marker: {
      color: '#64748b',
      size: 8,
      opacity: 0.5,
      line: { color: '#334155', width: 1 }
    }
  };

  const tracePareto = {
    x: paretoPoints.map((c: any) => c[xAxisKey]),
    y: paretoPoints.map((c: any) => c[yAxisKey]),
    customdata: paretoPoints.map((c: any) => c.id),
    text: paretoPoints.map((c: any) => 
      `<b>${c.name} ★ PARETO OPTIMAL</b><br>ID: ${c.id}<br>${xProp.label}: ${c[xAxisKey]}<br>${yProp.label}: ${c[yAxisKey]}<br>Toxicity Risk: ${Math.round(c.toxicity_risk_probability * 100)}%<br>Composite Score: ${c.composite_score}/100`
    ),
    hoverinfo: 'text',
    mode: 'markers',
    type: 'scatter',
    name: 'Pareto Optimal (Non-dominated)',
    marker: {
      color: '#ffffff',
      size: 13,
      symbol: 'diamond',
      opacity: 0.98,
      line: { color: '#38bdf8', width: 2 }
    }
  };

  const traceFrontierLine = {
    x: sortedPareto.map((c: any) => c[xAxisKey]),
    y: sortedPareto.map((c: any) => c[yAxisKey]),
    mode: 'lines',
    type: 'scatter',
    name: 'Frontier Envelope',
    line: {
      color: 'rgba(56, 189, 248, 0.55)',
      width: 2,
      dash: 'dot'
    },
    hoverinfo: 'none'
  };

  // 3D Plotly Traces
  const trace3DDominated = {
    x: dominatedPoints.map((c: any) => c.bioactivity_pic50),
    y: dominatedPoints.map((c: any) => c.solubility_logs),
    z: dominatedPoints.map((c: any) => c.toxicity_risk_probability),
    customdata: dominatedPoints.map((c: any) => c.id),
    text: dominatedPoints.map((c: any) => 
      `<b>${c.name}</b><br>pIC50: ${c.bioactivity_pic50}<br>logS: ${c.solubility_logs}<br>Tox: ${Math.round(c.toxicity_risk_probability * 100)}%`
    ),
    hoverinfo: 'text',
    mode: 'markers',
    type: 'scatter3d',
    name: 'Dominated Pool',
    marker: {
      color: '#64748b',
      size: 4,
      opacity: 0.45
    }
  };

  const trace3DPareto = {
    x: paretoPoints.map((c: any) => c.bioactivity_pic50),
    y: paretoPoints.map((c: any) => c.solubility_logs),
    z: paretoPoints.map((c: any) => c.toxicity_risk_probability),
    customdata: paretoPoints.map((c: any) => c.id),
    text: paretoPoints.map((c: any) => 
      `<b>${c.name} ★ PARETO</b><br>pIC50: ${c.bioactivity_pic50}<br>logS: ${c.solubility_logs}<br>Tox: ${Math.round(c.toxicity_risk_probability * 100)}%`
    ),
    hoverinfo: 'text',
    mode: 'markers',
    type: 'scatter3d',
    name: 'Pareto Front',
    marker: {
      color: '#ffffff',
      size: 7,
      symbol: 'diamond',
      opacity: 0.95,
      line: { color: '#38bdf8', width: 1.5 }
    }
  };

  const layout2D = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(8, 9, 14, 0.8)',
    margin: { l: 60, r: 30, t: 30, b: 60 },
    showlegend: true,
    legend: {
      font: { color: '#94a3b8', size: 11 },
      bgcolor: 'rgba(12, 14, 20, 0.85)',
      bordercolor: 'rgba(255, 255, 255, 0.1)',
      borderwidth: 1,
      x: 0.02,
      y: 0.98
    },
    xaxis: {
      title: { text: `${xProp.label} (${xProp.unit})`, font: { color: '#e2e8f0', size: 12 } },
      gridcolor: 'rgba(255, 255, 255, 0.06)',
      zerolinecolor: 'rgba(255, 255, 255, 0.12)',
      tickfont: { color: '#94a3b8' }
    },
    yaxis: {
      title: { text: `${yProp.label} (${yProp.unit})`, font: { color: '#e2e8f0', size: 12 } },
      gridcolor: 'rgba(255, 255, 255, 0.06)',
      zerolinecolor: 'rgba(255, 255, 255, 0.12)',
      tickfont: { color: '#94a3b8' }
    },
    autosize: true
  };

  const layout3D = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      xaxis: { title: 'Bioactivity (pIC50)', gridcolor: 'rgba(255,255,255,0.06)', color: '#e2e8f0', backgroundcolor: 'rgba(8, 9, 14, 0.8)' },
      yaxis: { title: 'Solubility (logS)', gridcolor: 'rgba(255,255,255,0.06)', color: '#e2e8f0', backgroundcolor: 'rgba(8, 9, 14, 0.8)' },
      zaxis: { title: 'Toxicity Risk (min)', gridcolor: 'rgba(255,255,255,0.06)', color: '#e2e8f0', backgroundcolor: 'rgba(8, 9, 14, 0.8)' }
    },
    showlegend: true,
    legend: {
      font: { color: '#94a3b8', size: 11 },
      bgcolor: 'rgba(12, 14, 20, 0.85)',
      x: 0.02,
      y: 0.98
    },
    autosize: true
  };

  const handlePlotClick = (eventData: any) => {
    if (!eventData || !eventData.points || eventData.points.length === 0) return;
    const pt = eventData.points[0];
    const candidateId = pt.customdata;
    if (candidateId) {
      const found = allCandidates.find((c: any) => c.id === candidateId);
      if (found) setSelectedCandidate(found);
      return;
    }
    const clickedName = pt.text?.split('<b>')[1]?.split('</b>')[0]?.replace(' ★ PARETO OPTIMAL', '')?.replace(' ★ PARETO', '')?.replace(' (Dominated)', '');
    const found = allCandidates.find((c: any) => c.name === clickedName);
    if (found) setSelectedCandidate(found);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Educational Header */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-3xl">
            <div className="flex items-center gap-2 text-xs font-mono tracking-wider uppercase text-slate-400">
              <GitCompare className="w-3.5 h-3.5 text-cyan-400" />
              <span>Stage 04 • Multi-Objective Pareto Laboratory</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-medium tracking-tight text-white">
              Pareto Frontier & Multi-Property Trade-Off Navigation
            </h2>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-normal">
              In medicinal discovery, no compound possesses maximal potency, ideal aqueous solubility, zero toxicity, and optimal clearance simultaneously. A <ScientificTooltip id="pareto"><span className="text-white underline decoration-dotted cursor-help">Pareto-optimal solution</span></ScientificTooltip> is a candidate for which <strong>no other candidate in the library is superior across every single property</strong>.
            </p>
          </div>

          <div className="flex flex-wrap lg:flex-col gap-2 shrink-0">
            <div className="px-4 py-2 rounded-2xl bg-black/40 border border-white/[0.08] text-xs font-mono">
              <span className="text-slate-400">Candidate Pool:</span>
              <strong className="text-white ml-2">{data.total_candidates} Compounds</strong>
            </div>
            <div className="px-4 py-2 rounded-2xl bg-white/[0.04] border border-white/[0.1] text-xs font-mono">
              <span className="text-slate-300">Pareto Frontier:</span>
              <strong className="text-white ml-2">{data.pareto_optimal_count} Optimal Leads</strong>
            </div>
          </div>
        </div>

        {/* Trade-off summary pills */}
        <div className="mt-6 pt-5 border-t border-white/[0.06] grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="flex items-start gap-3 bg-black/30 p-3.5 rounded-2xl border border-white/[0.06]">
            <Award className="w-4 h-4 text-white shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Pareto Diamonds (White):</strong>
              <p className="text-slate-400 text-[11px] mt-0.5 font-sans">Non-dominated leads representing optimal biological trade-offs.</p>
            </div>
          </div>
          <div className="flex items-start gap-3 bg-black/30 p-3.5 rounded-2xl border border-white/[0.06]">
            <span className="w-3.5 h-3.5 rounded-full bg-slate-500 shrink-0 mt-1" />
            <div>
              <strong className="text-white">Dominated Points (Slate):</strong>
              <p className="text-slate-400 text-[11px] mt-0.5 font-sans">Compounds strictly beaten by at least one other molecule across all criteria.</p>
            </div>
          </div>
          <div className="flex items-start gap-3 bg-black/30 p-3.5 rounded-2xl border border-white/[0.06]">
            <Sparkles className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <strong className="text-white">Composite Score [0-100]:</strong>
              <p className="text-slate-400 text-[11px] mt-0.5 font-sans">Normalized desirability composite evaluating affinity, safety, and solubility.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Interactive Plot & Selected Candidate Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left: Plotly Visualization Canvas */}
        <div className="lg:col-span-8 bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl flex flex-col justify-between">
          <div>
            {/* Plot Controls & Projection Selectors */}
            <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-white/[0.06]">
              <div className="flex items-center gap-2.5">
                <TrendingUp className="w-5 h-5 text-slate-300" />
                <div>
                  <h3 className="font-medium text-white text-base">Trade-off Landscape</h3>
                  <p className="text-xs text-slate-400 font-mono">
                    {plotMode === '2d' ? '2D Projection of Multi-Objective Space' : '3D Interactive Multi-Property Space'}
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                {/* 2D / 3D Mode Switcher */}
                <div className="flex bg-black/40 border border-white/[0.08] rounded-full p-1 text-xs">
                  <button
                    onClick={() => setPlotMode('2d')}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-all cursor-pointer ${
                      plotMode === '2d' ? 'bg-white text-zinc-950 font-semibold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    2D Scatter
                  </button>
                  <button
                    onClick={() => setPlotMode('3d')}
                    className={`px-3.5 py-1.5 rounded-full font-medium transition-all cursor-pointer ${
                      plotMode === '3d' ? 'bg-white text-zinc-950 font-semibold shadow' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    3D Multi-Cube
                  </button>
                </div>

                {/* Axis Selectors for 2D Mode */}
                {plotMode === '2d' && (
                  <div className="flex items-center gap-2 text-xs">
                    <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-full border border-white/[0.08]">
                      <span className="text-slate-400 font-mono">X:</span>
                      <select
                        value={xAxisKey}
                        onChange={(e) => setXAxisKey(e.target.value)}
                        className="bg-transparent text-white text-xs focus:outline-none cursor-pointer"
                      >
                        {PROPERTY_OPTIONS.map((p) => (
                          <option key={p.key} value={p.key} className="bg-zinc-900 text-white">{p.label}</option>
                        ))}
                      </select>
                    </div>

                    <div className="flex items-center gap-1.5 bg-black/40 px-3 py-1.5 rounded-full border border-white/[0.08]">
                      <span className="text-slate-400 font-mono">Y:</span>
                      <select
                        value={yAxisKey}
                        onChange={(e) => setYAxisKey(e.target.value)}
                        className="bg-transparent text-white text-xs focus:outline-none cursor-pointer"
                      >
                        {PROPERTY_OPTIONS.map((p) => (
                          <option key={p.key} value={p.key} className="bg-zinc-900 text-white">{p.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Plotly Canvas Container */}
            <div className="w-full h-[420px] rounded-2xl overflow-hidden bg-black/40 border border-white/[0.06]">
              {plotMode === '2d' ? (
                <Plot
                  data={[traceDominated, tracePareto, traceFrontierLine]}
                  layout={layout2D}
                  onClick={handlePlotClick}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%', height: '100%' }}
                />
              ) : (
                <Plot
                  data={[trace3DDominated, trace3DPareto]}
                  layout={layout3D}
                  onClick={handlePlotClick}
                  config={{ responsive: true, displayModeBar: false }}
                  style={{ width: '100%', height: '100%' }}
                />
              )}
            </div>
          </div>

          <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs text-slate-400 gap-2 font-mono">
            <span>Click any point on the canvas to inspect structure and trade-offs.</span>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-white">
                <span className="w-2.5 h-2.5 rounded-sm bg-white inline-block border border-cyan-400" /> Pareto Optimal ({paretoPoints.length})
              </span>
              <span className="flex items-center gap-1.5 text-slate-400">
                <span className="w-2.5 h-2.5 rounded-sm bg-slate-500 inline-block" /> Dominated ({dominatedPoints.length})
              </span>
            </div>
          </div>
        </div>

        {/* Right: Selected Candidate Inspection Card */}
        <div className="lg:col-span-4 bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-white/[0.06]">
              <h4 className="font-medium text-white text-sm flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-slate-300" /> Candidate Inspection
              </h4>
              {selectedCandidate?.is_pareto_optimal ? (
                <span className="px-3 py-0.5 rounded-full text-xs font-mono font-semibold bg-white/[0.06] text-white border border-white/20 flex items-center gap-1">
                  <Award className="w-3 h-3 text-cyan-400" /> Pareto Front
                </span>
              ) : (
                <span className="px-3 py-0.5 rounded-full text-xs font-mono font-medium bg-black/40 text-slate-400 border border-white/[0.06]">
                  Dominated (Rank {selectedCandidate?.pareto_rank})
                </span>
              )}
            </div>

            {selectedCandidate ? (
              <div className="space-y-4">
                {/* 2D Structure Preview */}
                <div className="bg-black/40 rounded-2xl p-4 border border-white/[0.06] flex items-center justify-center min-h-[160px]">
                  <div 
                    className="w-full flex justify-center"
                    dangerouslySetInnerHTML={{ __html: selectedCandidate.svg_2d }}
                  />
                </div>

                {/* Candidate Name & Identifier */}
                <div>
                  <div className="flex items-baseline justify-between">
                    <h5 className="text-base font-medium text-white">{selectedCandidate.name}</h5>
                    <span className="text-xs font-mono text-white font-semibold">
                      Score: {selectedCandidate.composite_score}/100
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">{selectedCandidate.id}</div>
                  <div className="text-[11px] font-mono text-slate-300 truncate mt-1.5 bg-black/40 p-2 rounded-xl border border-white/[0.06]">
                    {selectedCandidate.smiles}
                  </div>
                </div>

                {/* Decision Rationale */}
                <div className="p-3.5 bg-black/40 rounded-2xl border border-white/[0.06] text-xs">
                  <div className="font-medium text-white mb-1 flex items-center gap-1.5">
                    <Info className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Decision Rationale:</span>
                  </div>
                  <p className="text-slate-300 text-[11px] leading-relaxed">
                    {selectedCandidate.is_pareto_optimal ? (
                      <>
                        <strong>Non-dominated lead:</strong> Exhibits high binding potency (pIC50 = {selectedCandidate.bioactivity_pic50}) without exceeding clinical toxicity thresholds ({Math.round(selectedCandidate.toxicity_risk_probability * 100)}%). Cannot be replaced without compromising either solubility or affinity.
                      </>
                    ) : (
                      <>
                        <strong>Sub-optimal lead:</strong> Dominated by Rank 1 compounds that achieve superior bioactivity and solubility at equivalent or lower toxicity risk.
                      </>
                    )}
                  </p>
                </div>

                {/* Property Grid */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-black/40 p-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">Bioactivity pIC50</span>
                    <div className="text-base font-mono text-cyan-300 mt-0.5">
                      {selectedCandidate.bioactivity_pic50}
                    </div>
                  </div>
                  <div className="bg-black/40 p-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">Solubility logS</span>
                    <div className="text-base font-mono text-white mt-0.5">
                      {selectedCandidate.solubility_logs}
                    </div>
                  </div>
                  <div className="bg-black/40 p-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">Toxicity Risk</span>
                    <div className={`text-base font-mono mt-0.5 ${
                      selectedCandidate.toxicity_risk_probability < 0.25 ? 'text-emerald-400' : 'text-amber-400'
                    }`}>
                      {Math.round(selectedCandidate.toxicity_risk_probability * 100)}%
                    </div>
                  </div>
                  <div className="bg-black/40 p-3 rounded-xl border border-white/[0.06]">
                    <span className="text-slate-400 font-mono">QED Score</span>
                    <div className="text-base font-mono text-emerald-400 mt-0.5">
                      {selectedCandidate.qed_drug_likeness}
                    </div>
                  </div>
                </div>

                {/* AD status */}
                <div className="flex items-center justify-between p-3 bg-black/40 rounded-xl border border-white/[0.06] text-xs font-mono">
                  <span className="text-slate-400">Applicability Domain:</span>
                  <span className={`font-semibold ${
                    selectedCandidate.ad_status === 'IN_DOMAIN' ? 'text-emerald-400' : 'text-amber-400'
                  }`}>
                    {selectedCandidate.ad_status} ({Math.round(selectedCandidate.ad_score * 100)}%)
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-400 text-center py-10 font-mono">Select a candidate to view full trade-off metrics.</p>
            )}
          </div>

          {selectedCandidate && onSelectCandidate && (
            <button
              onClick={() => onSelectCandidate(selectedCandidate.smiles)}
              className="mt-6 w-full py-3 bg-white hover:bg-slate-200 text-zinc-950 font-semibold text-xs rounded-full shadow-lg transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              <span>Examine Structure in Stage 01</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Candidate Matrix Table with Search & Filter */}
      <div className="bg-[#0c0d14]/70 border border-white/[0.08] rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-2xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-white/[0.06]">
          <div>
            <h3 className="font-medium text-white text-base">Candidate Comparison Matrix</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Ranked by Pareto non-domination level and composite desirability.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 w-full sm:w-auto">
            {/* Search Input */}
            <div className="relative flex-1 sm:w-60">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search candidates..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-black/40 border border-white/[0.08] rounded-full pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-white/30"
              />
            </div>

            {/* Pareto Only Toggle */}
            <button
              onClick={() => setFilterParetoOnly(!filterParetoOnly)}
              className={`px-4 py-2 rounded-full text-xs font-mono font-medium flex items-center gap-1.5 transition-all cursor-pointer shrink-0 ${
                filterParetoOnly 
                  ? 'bg-white text-zinc-950 font-semibold shadow' 
                  : 'bg-white/[0.04] text-slate-300 hover:text-white border border-white/[0.08]'
              }`}
            >
              <Filter className="w-3.5 h-3.5" />
              {filterParetoOnly ? 'Pareto Only' : 'Show All'}
            </button>
          </div>
        </div>

        {/* Responsive Table */}
        <div className="overflow-x-auto rounded-2xl border border-white/[0.06]">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-black/50 text-slate-400 font-medium uppercase tracking-wider text-[11px] font-mono border-b border-white/[0.06]">
              <tr>
                <th className="py-3 px-4">Compound</th>
                <th className="py-3 px-4">Pareto Status</th>
                <th className="py-3 px-4">Composite</th>
                <th className="py-3 px-4">Bioactivity</th>
                <th className="py-3 px-4">Solubility</th>
                <th className="py-3 px-4">Tox Risk</th>
                <th className="py-3 px-4">QED</th>
                <th className="py-3 px-4">AD Status</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] bg-black/20 font-mono">
              {displayedCandidates.map((c: any) => {
                const isSelected = selectedCandidate?.id === c.id;
                return (
                  <tr 
                    key={c.id} 
                    onClick={() => setSelectedCandidate(c)}
                    className={`hover:bg-white/[0.03] cursor-pointer transition-colors ${
                      isSelected ? 'bg-white/[0.06] border-l-2 border-white' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-white">
                      <div>{c.name}</div>
                      <div className="text-[10px] text-slate-400 font-mono">{c.id}</div>
                    </td>
                    <td className="py-3 px-4">
                      {c.is_pareto_optimal ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-white font-medium bg-white/[0.06] px-2.5 py-0.5 rounded-full border border-white/[0.1]">
                          <Award className="w-3 h-3 text-cyan-400" /> Pareto Front
                        </span>
                      ) : (
                        <span className="text-[11px] text-slate-400">Rank {c.pareto_rank}</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-medium text-white">
                      {c.composite_score}
                    </td>
                    <td className="py-3 px-4 text-cyan-300">
                      {c.bioactivity_pic50}
                    </td>
                    <td className="py-3 px-4 text-slate-200">
                      {c.solubility_logs} <span className="text-[10px] text-slate-400">[{c.solubility_lower_90}, {c.solubility_upper_90}]</span>
                    </td>
                    <td className="py-3 px-4">
                      <span className={c.toxicity_risk_probability < 0.25 ? 'text-emerald-400' : 'text-amber-400'}>
                        {Math.round(c.toxicity_risk_probability * 100)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-emerald-400">
                      {c.qed_drug_likeness}
                    </td>
                    <td className="py-3 px-4 text-[11px]">
                      <span className={`px-2 py-0.5 rounded-full ${
                        c.ad_status === 'IN_DOMAIN' ? 'text-emerald-400 bg-emerald-500/10' : 'text-amber-400 bg-amber-500/10'
                      }`}>
                        {c.ad_status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-sans">
                      {onSelectCandidate && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectCandidate(c.smiles);
                          }}
                          className="px-3 py-1 bg-white/[0.05] hover:bg-white hover:text-zinc-950 text-slate-300 rounded-full text-xs transition-all cursor-pointer font-medium"
                        >
                          Inspect
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Expandable Section: Mathematical Dominance */}
        <div className="mt-5 pt-4 border-t border-white/[0.06]">
          <button
            onClick={() => setShowMathDetails(!showMathDetails)}
            className="w-full py-2 px-4 rounded-xl bg-black/40 hover:bg-white/[0.04] text-slate-300 hover:text-white text-xs font-mono flex items-center justify-between transition-colors border border-white/[0.06] cursor-pointer"
          >
            <span className="flex items-center gap-2">
              <Info className="w-3.5 h-3.5 text-slate-300" />
              <span>{showMathDetails ? 'Collapse Pareto Formalism' : 'Expand Pareto Non-Domination Mathematical Formulation'}</span>
            </span>
            {showMathDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showMathDetails && (
            <div className="mt-3 p-5 rounded-2xl bg-black/40 border border-white/[0.06] text-xs space-y-3 font-mono">
              <div className="text-xs font-semibold text-white">
                Formal Definition of Pareto Dominance
              </div>
              <p className="text-slate-300 leading-relaxed text-[11px]">
                A candidate vector x(1) Pareto-dominates candidate vector x(2) (denoted x(1) ≺ x(2)) if and only if:
              </p>
              <div className="p-3 bg-white/[0.03] border border-white/[0.06] rounded-xl text-white text-center">
                {"∀ i ∈ {1,...,M}: f_i(x(1)) ≥ f_i(x(2))  ∧  ∃ j ∈ {1,...,M}: f_j(x(1)) > f_j(x(2))"}
              </div>
              <p className="text-slate-300 leading-relaxed text-[11px]">
                Non-dominated sorting partitions candidates into concentric Pareto shells. Rank 1 solutions form the frontier where further improvement in one objective mandates sacrificing another.
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};
