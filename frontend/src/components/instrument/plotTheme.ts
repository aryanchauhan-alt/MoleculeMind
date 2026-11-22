/**
 * Shared Plotly theme.
 *
 * Charts are the one place where a design system usually falls apart, because
 * plotting libraries ship their own opinions. Everything here is defined once
 * so every chart in the app reads as part of the same instrument.
 */

export const PLOT_COLORS = {
  base: '#F4F3EF',
  surface: '#FFFFFF',
  line: 'rgba(23, 23, 26, 0.09)',
  text: '#17171A',
  dim: '#6E6E76',
  faint: '#9B9BA2',
  signal: '#2B2BE0',
  caution: '#B4530F',
  alarm: '#B3261E',
} as const;

const axis = (title?: string) => {
  const a: Record<string, unknown> = {
    gridcolor: PLOT_COLORS.line,
    zerolinecolor: 'rgba(23,23,26,0.16)',
    color: PLOT_COLORS.dim,
    tickfont: { size: 10, family: 'JetBrains Mono, monospace', color: PLOT_COLORS.faint },
    linecolor: PLOT_COLORS.line,
  };
  // Only set `title` when there is one. Plotly treats an explicit `undefined`
  // differently from an absent key and can fail to render at all.
  if (title) a.title = { text: title, font: { size: 11, color: PLOT_COLORS.dim } };
  return a;
};

export const plotLayout = (
  opts: { x?: string; y?: string; z?: string; legend?: boolean; height?: number } = {}
): Record<string, unknown> => {
  const layout: Record<string, unknown> = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: '#FFFFFF',
    font: { family: 'JetBrains Mono, monospace', size: 11, color: PLOT_COLORS.text },
    margin: { l: 62, r: 24, t: 18, b: 52 },
    autosize: true,
    xaxis: axis(opts.x),
    yaxis: axis(opts.y),
    showlegend: opts.legend ?? false,
    legend: {
      orientation: 'h',
      y: -0.2,
      font: { size: 10, color: PLOT_COLORS.dim },
      bgcolor: 'rgba(0,0,0,0)',
    },
    hoverlabel: {
      bgcolor: '#FFFFFF',
      bordercolor: 'rgba(23,23,26,0.18)',
      font: { family: 'JetBrains Mono, monospace', size: 11, color: PLOT_COLORS.text },
    },
  };

  // Same reasoning: omit these keys entirely rather than setting them undefined.
  if (opts.z) {
    layout.scene = {
      xaxis: { ...axis(opts.x), backgroundcolor: '#FFFFFF' },
      yaxis: { ...axis(opts.y), backgroundcolor: '#FFFFFF' },
      zaxis: { ...axis(opts.z), backgroundcolor: '#FFFFFF' },
    };
  }
  if (opts.height) layout.height = opts.height;

  return layout;
};

export const PLOT_CONFIG = { displayModeBar: false, responsive: true } as const;
