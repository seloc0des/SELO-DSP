import React, { useEffect, useMemo, useRef, useState } from 'react';
import { formatTraitNameLowercase } from '../../utils/formatTraitName';

/**
 * PersonaTraitRadar (read-only)
 * Displays trait weights on a radar/spider chart using pure SVG.
 * traits: Array<{ name, value: 0..1, category?: string }>
 */
const PersonaTraitRadar = ({ traits = [], size = 400, levels = 4, scaleLabel = 'Scale: 0.0–1.0', padding: paddingProp }) => {
  // Responsive: observe container width and cap the drawing size
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(size);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === 'undefined') {
      return;
    }
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = Math.floor(entry.contentRect.width);
        if (w && w !== containerWidth) setContainerWidth(w);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [containerWidth]);

  const effectiveSize = Math.min(size, containerWidth || size);

  const { points, labels, maxRadius } = useMemo(() => {
    const valid = traits.filter(t => typeof t.value === 'number');
    const n = valid.length;
    const cx = effectiveSize / 2;
    const cy = effectiveSize / 2;

    // Estimate label width to determine safe padding so labels stay inside the SVG viewport.
    const maxLabelLen = valid.reduce((m, t) => {
      const name = typeof t.name === 'string' ? t.name : '';
      return Math.max(m, name.length);
    }, 0);
    // Approx 7.5px per char at 13px font; clamp to reasonable range
    const estLabelPx = Math.min(Math.max(maxLabelLen * 7.5, 48), Math.floor(effectiveSize * 0.45));
    const padding = typeof paddingProp === 'number' ? paddingProp : Math.max(60, estLabelPx);
    const r = Math.max(12, (effectiveSize / 2) - padding);

    const angleFor = (i) => (Math.PI * 2 * i) / n - Math.PI / 2; // start at top
    const toPoint = (i, magnitude) => {
      const ang = angleFor(i);
      return [cx + Math.cos(ang) * r * magnitude, cy + Math.sin(ang) * r * magnitude];
    };

    const pts = valid.map((t, i) => toPoint(i, Math.max(0, Math.min(1, t.value))));
    const lbls = valid.map((t, i) => ({
      name: formatTraitNameLowercase(t.name),
      category: t.category || 'general',
      anchor: (() => {
        const ang = angleFor(i);
        const x = cx + Math.cos(ang) * (r + 18);
        const y = cy + Math.sin(ang) * (r + 18);
        let anchor = 'middle';
        if (Math.cos(ang) > 0.3) anchor = 'start';
        else if (Math.cos(ang) < -0.3) anchor = 'end';
        return { x, y, anchor };
      })(),
    }));

    return { points: pts, labels: lbls, maxRadius: r };
  }, [traits, effectiveSize, paddingProp]);

  if (!traits.length) {
    return <div className="text-gray-500 text-xs italic">No traits defined</div>;
  }

  const cx = effectiveSize / 2;
  const cy = effectiveSize / 2;

  const polygon = points.map(([x, y]) => `${x},${y}`).join(' ');

  return (
    <div ref={containerRef} className="w-full flex flex-col items-center">
      <svg width={effectiveSize} height={effectiveSize} viewBox={`0 0 ${effectiveSize} ${effectiveSize}`} role="img" aria-label="Persona trait radar chart">
        {/* Rings */}
        {[...Array(levels)].map((_, i) => {
          const ratio = (i + 1) / levels;
          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={maxRadius * ratio}
              className="fill-none stroke-[var(--color-border)]"
              strokeOpacity={0.5}
              strokeWidth={1}
            />
          );
        })}

        {/* Axes */}
        {points.map(([x, y], i) => (
          <line key={i} x1={cx} y1={cy} x2={x} y2={y} className="stroke-[var(--color-border)]" strokeOpacity={0.5} strokeWidth={1} />
        ))}

        {/* Filled polygon */}
        <polygon points={polygon} className="fill-[var(--color-accent)]/20 stroke-[var(--color-accent)]" strokeWidth={2} />

        {/* Vertices */}
        {points.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={3} className="fill-[var(--color-accent-strong)] stroke-[var(--color-accent-strong)]" />
        ))}

        {/* Labels */}
        {labels.map((l, i) => (
          <text
            key={i}
            x={l.anchor.x}
            y={l.anchor.y}
            textAnchor={l.anchor.anchor}
            dominantBaseline="middle"
            className="fill-[var(--color-text-secondary)] text-[13px] font-medium"
          >
            {l.name}
          </text>
        ))}
      </svg>
      <div className="text-[11px] text-[var(--color-text-muted)] mt-1" aria-hidden>{scaleLabel}</div>
      <div className="text-xs text-gray-500 mt-1">Read‑only visualization. Traits evolve autonomously.</div>
    </div>
  );
};

export default PersonaTraitRadar;
