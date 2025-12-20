import React, { useState, useEffect } from 'react';
import personaService from '../../services/personaService';

export const parseJsonFromString = (value) => {
  if (typeof value !== 'string') return null;
  let trimmed = value.trim();
  if (!trimmed) return null;

  const stripFence = (str) => {
    const fenceStart = str.match(/^```(?:json)?/i);
    if (fenceStart) {
      str = str.replace(/^```(?:json)?/i, '');
    }
    if (str.endsWith('```')) {
      str = str.slice(0, -3);
    }
    return str.trim();
  };

  const sanitiseLooseJson = (str) => {
    let candidate = stripFence(str);
    if (!candidate) return '';
    candidate = candidate.replace(/"intensity"\s*:\s*,/g, '"intensity": null,');
    candidate = candidate.replace(/,\s*(?=[}\]])/g, '');
    return candidate.trim();
  };

  const tryParse = (str) => {
    if (!str) return null;
    try {
      return JSON.parse(str);
    } catch (e) {
      return null;
    }
  };

  let candidate = sanitiseLooseJson(trimmed);
  let parsed = tryParse(candidate);
  if (parsed) return parsed;

  const firstBrace = candidate.indexOf('{');
  const lastBrace = candidate.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    parsed = tryParse(candidate.slice(firstBrace, lastBrace + 1));
    if (parsed) return parsed;
  }

  return null;
};

export const sanitizeContentText = (value) => {
  if (typeof value !== 'string') return '';
  const trimmed = value.trim();
  if (!trimmed) return '';
  const parsed = parseJsonFromString(trimmed);
  if (parsed && typeof parsed.content === 'string' && parsed.content.trim()) {
    return parsed.content.trim();
  }
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  if (fenced) {
    return fenced[1].trim();
  }
  return trimmed.replace(/^```(?:json)?/i, '').replace(/```$/i, '').trim();
};

export const toArray = (value) => {
  if (Array.isArray(value)) {
    return value.filter((item) => item !== null && item !== undefined && item !== '');
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  return [];
};

export const normalizeTraitChanges = (value) => {
  if (!Array.isArray(value)) return [];
  return value.filter((item) => item && typeof item === 'object');
};

const formatDisplayLabel = (value) => {
  if (typeof value !== 'string') return value;
  const cleaned = value.replace(/[_\s]+/g, ' ').trim();
  if (!cleaned) return '';
  return cleaned.replace(/\b\w/g, (c) => c.toUpperCase());
};

/**
 * ReflectionCard displays an individual reflection with expandable content
 */
const ReflectionCard = ({ reflection, timeAgo }) => {
  const reflectionId = reflection?.id || reflection?.reflection_id;
  
  // Load expanded state from localStorage
  const getExpandedState = () => {
    try {
      if (!reflectionId) return false;
      const key = `reflection_expanded_${reflectionId}`;
      const stored = localStorage.getItem(key);
      return stored === 'true';
    } catch (e) {
      return false;
    }
  };

  const [expanded, setExpanded] = useState(getExpandedState);
  // State for traits
  const [topTraits, setTopTraits] = useState([]);
  const [loadingTraits, setLoadingTraits] = useState(false);

  // Persist expanded state to localStorage
  useEffect(() => {
    if (reflectionId) {
      try {
        const key = `reflection_expanded_${reflectionId}`;
        localStorage.setItem(key, expanded.toString());
      } catch (e) {
        console.warn('Failed to save expanded state:', e);
      }
    }
  }, [expanded, reflectionId]);

  // Fetch traits when component mounts
  useEffect(() => {
    let mounted = true;
    
    const fetchTraits = async () => {
      try {
        setLoadingTraits(true);
        // Fetch default persona traits
        const resp = await personaService.getDefaultPersonaTraits();
        if (!mounted) return;
        
        // Normalize to { name, value, category }
        const traits = Array.isArray(resp?.traits) ? resp.traits : [];
        const normalized = traits
          .map(t => ({
            name: t?.name || t?.trait || t?.id || '',
            value: typeof t?.value === 'number' ? t.value : 
                   (typeof t?.weight === 'number' ? t.weight : null),
            category: t?.category || '',
          }))
          .filter(t => t.name && t.value !== null)
          .sort((a, b) => (b.value - a.value))
          .slice(0, 6);
          
        if (mounted) setTopTraits(normalized);
      } catch (e) {
        if (mounted) setTopTraits([]);
      } finally {
        if (mounted) setLoadingTraits(false);
      }
    };
    
    fetchTraits();
    return () => { mounted = false; };
  }, []);

  // Normalize result structure (support string, object, or missing)
  let parsedResult = {};
  try {
    if (typeof reflection?.result === 'string') {
      parsedResult = JSON.parse(reflection.result);
    } else if (typeof reflection?.result === 'object' && reflection?.result !== null) {
      parsedResult = reflection.result;
    } else {
      parsedResult = {};
    }
  } catch (e) {
    parsedResult = {};
  }

  const fallbackParsed = parseJsonFromString(reflection?.content);
  if (fallbackParsed) {
    parsedResult = { ...fallbackParsed, ...parsedResult };
  }

  const pickList = (...candidates) => {
    for (const candidate of candidates) {
      const normalized = toArray(candidate);
      if (normalized.length > 0) {
        return normalized;
      }
    }
    return [];
  };

  const rawContent = parsedResult.content || reflection.content || '';
  const content = sanitizeContentText(rawContent);
  const themes = pickList(parsedResult.themes, reflection.themes);
  const insights = pickList(parsedResult.insights, reflection.insights);
  const actions = pickList(parsedResult.actions, reflection.actions);
  const trait_changes = normalizeTraitChanges(
    parsedResult.trait_changes ?? reflection.trait_changes
  );

  const baseEmotionalState = {
    ...(reflection?.emotional_state && typeof reflection.emotional_state === 'object'
      ? reflection.emotional_state
      : {}),
    ...(parsedResult.emotional_state && typeof parsedResult.emotional_state === 'object'
      ? parsedResult.emotional_state
      : {}),
  };
  const secondary = toArray(baseEmotionalState.secondary);
  let intensity = baseEmotionalState.intensity;
  if (typeof intensity === 'string') {
    const parsed = parseFloat(intensity);
    intensity = Number.isFinite(parsed) ? parsed : undefined;
  }
  if (typeof intensity === 'number') {
    intensity = Math.max(0, Math.min(1, intensity));
  }
  const emotional_state = {
    ...baseEmotionalState,
    ...(typeof intensity === 'number' ? { intensity } : {}),
    ...(secondary.length > 0 ? { secondary } : {}),
  };

  return (
    <div className="border border-[var(--color-border)] bg-[var(--color-bg-elev-1)] rounded-lg overflow-hidden transition-all">
      {/* Card Header */}
      <div className="p-4 flex justify-between items-center">
        <div>
          <h4 className="text-lg font-medium text-[var(--color-accent)]">
            {(reflection.reflection_type || reflection.type || 'reflection')
              .toString()
              .replace(/^(.)/, (m, c) => c.toUpperCase())} Reflection
          </h4>
          <p className="text-xs text-gray-400">{timeAgo}</p>
        </div>
        
        <button 
          onClick={() => setExpanded(!expanded)} 
          className="text-[var(--color-accent-strong)] hover:opacity-90 transition-colors"
          aria-expanded={expanded}
        >
          <span className="sr-only">{expanded ? 'Collapse' : 'Expand'}</span>
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className={`h-5 w-5 transition-transform ${expanded ? 'rotate-180' : ''}`}
            viewBox="0 0 20 20" 
            fill="currentColor"
          >
            <path 
              fillRule="evenodd" 
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" 
              clipRule="evenodd" 
            />
          </svg>
        </button>
      </div>
      
      {/* Preview (always visible) */}
      <div className="px-4 pb-2">
        <div className="flex flex-wrap gap-2">
          {themes.map((theme, i) => (
            <span 
              key={i} 
              className="px-2 py-1 text-xs rounded-full bg-[color:var(--color-accent)/0.08] text-[var(--color-accent)] border border-[var(--color-border)]"
            >
              {formatDisplayLabel(theme)}
            </span>
          ))}
        </div>
      </div>
      
      {/* Expanded content */}
      {expanded && (
        <div className="p-4 border-t border-[var(--color-border)] mt-2 bg-[var(--color-bg-elev-1)]">
          {/* Main reflection content */}
          <div className="space-y-4">
            <div className="prose prose-sm prose-invert max-w-none whitespace-pre-wrap">{content}</div>
            
            {/* Trait Changes - Shows evolution during this reflection */}
            {trait_changes.length > 0 && (
              <div className="mt-4 p-3 rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
                <h5 className="text-sm font-medium text-[var(--color-accent-strong)] mb-2">📊 Trait Changes</h5>
                <div className="space-y-1">
                  {trait_changes.map((change, i) => {
                    const traitName = formatDisplayLabel(change.trait || change.name || 'Unknown');
                    const delta = change.delta || change.change || 0;
                    const oldVal = change.old_value || change.from;
                    const newVal = change.new_value || change.to;
                    const isIncrease = delta > 0;
                    const isDecrease = delta < 0;
                    
                    return (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-[var(--color-text-secondary)]">{traitName}</span>
                        <div className="flex items-center gap-2">
                          {oldVal !== undefined && newVal !== undefined && (
                            <span className="text-[var(--color-text-muted)]">
                              {typeof oldVal === 'number' ? oldVal.toFixed(2) : oldVal} → {typeof newVal === 'number' ? newVal.toFixed(2) : newVal}
                            </span>
                          )}
                          <span className={`font-medium ${isIncrease ? 'text-green-400' : isDecrease ? 'text-red-400' : 'text-gray-400'}`}>
                            {isIncrease ? '↑' : isDecrease ? '↓' : '='} {Math.abs(delta).toFixed(3)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            
            {/* Current Traits (moved from preview to expanded section) */}
            {topTraits.length > 0 && (
              <div className="mt-4 p-3 rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
                <h5 className="text-sm font-medium text-[var(--color-accent-strong)] mb-2">🏷️ Current Traits</h5>
                <div className="flex flex-wrap gap-2">
                  {topTraits.map((t, i) => (
                    <span
                      key={`${t.name}-${i}`}
                      title={t.category ? `${t.name} • ${t.category}` : t.name}
                      className="px-2 py-0.5 text-[10px] rounded-full bg-[var(--color-bg-elev-2)] text-[var(--color-text-secondary)] border border-[var(--color-border)]"
                    >
                      {formatDisplayLabel(t.name)}
                      {typeof t.value === 'number' && (
                        <>
                          {" "}
                          <span className="text-[var(--color-accent)]">{(t.value).toFixed(2)}</span>
                        </>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {/* Emotional state */}
            {emotional_state && Object.keys(emotional_state).length > 0 && (
              <div className="mt-4 p-3 rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
                <h5 className="text-sm font-medium text-[var(--color-accent-strong)] mb-2">Emotional State</h5>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {emotional_state.primary && (
                    <div>
                      <span className="text-[var(--color-text-muted)]">Primary:</span>{" "}
                      <span className="text-[var(--color-text-primary)]">{formatDisplayLabel(emotional_state.primary)}</span>
                    </div>
                  )}
                  {emotional_state.intensity !== undefined && (
                    <div>
                      <span className="text-[var(--color-text-muted)]">Intensity:</span>{" "}
                      <span className="text-[var(--color-text-primary)]">{(emotional_state.intensity * 100).toFixed(0)}%</span>
                    </div>
                  )}
                  {emotional_state.secondary?.length > 0 && (
                    <div className="col-span-2">
                      <span className="text-[var(--color-text-muted)]">Secondary:</span>{" "}
                      <span className="text-[var(--color-text-primary)]">{emotional_state.secondary.map(formatDisplayLabel).join(', ')}</span>
                    </div>
                  )}
                  {emotional_state.trend && (
                    <div className="col-span-2">
                      <span className="text-[var(--color-text-muted)]">Trend:</span>{" "}
                      <span className="text-[var(--color-text-primary)]">{formatDisplayLabel(emotional_state.trend)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* Insights and Actions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              {insights.length > 0 && (
                <div className="p-3 rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
                  <h5 className="text-sm font-medium text-[var(--color-accent)] mb-2">Key Insights</h5>
                  <ul className="list-disc list-inside text-xs space-y-1 text-[var(--color-text-secondary)]">
                    {insights.map((insight, i) => (
                      <li key={i}>{insight}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {actions.length > 0 && (
                <div className="p-3 rounded-md bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
                  <h5 className="text-sm font-medium text-[var(--color-accent-strong)] mb-2">Suggested Actions</h5>
                  <ul className="list-disc list-inside text-xs space-y-1 text-[var(--color-text-secondary)]">
                    {actions.map((action, i) => (
                      <li key={i}>{action}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Card Footer */}
      <div className="p-3 bg-[var(--color-bg-elev-1)] border-t border-[var(--color-border)] flex justify-between items-center">
        {reflection.id && (
          <span className="text-xs text-[var(--color-text-muted)]">
            ID: {reflection.id.substring(0, 8)}
          </span>
        )}
        
        <button 
          onClick={() => setExpanded(!expanded)} 
          className="text-xs text-[var(--color-accent)] hover:opacity-90 transition-colors"
        >
          {expanded ? 'Show Less' : 'Show More'}
        </button>
      </div>
    </div>
  );
};

export default ReflectionCard;
