import React, { useEffect, useMemo, useState } from 'react';
import personaService from '../../services/personaService';
import PersonaTraitRadar from './PersonaTraitRadar';
import { personaLogger as logger } from '../../utils/logger';
import { formatTraitName } from '../../utils/formatTraitName';

/**
 * PersonaAnalytics
 * - Shows current default persona traits (read-only)
 * - On trait select, fetches and displays short history list
 */
const PersonaAnalytics = () => {
  const [traits, setTraits] = useState([]);
  const [selectedTrait, setSelectedTrait] = useState(null);
  const [history, setHistory] = useState([]);
  const [personaId, setPersonaId] = useState(null);
  const [loadingTraits, setLoadingTraits] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [error, setError] = useState(null);
  const [lastAction, setLastAction] = useState(null); // 'loadTraits' | 'loadHistory'

  useEffect(() => {
    const load = async () => {
      try {
        setLoadingTraits(true);
        setError(null);
        setLastAction('loadTraits');
        const res = await personaService.getDefaultPersonaTraits();
        const items = res?.data?.traits || res?.traits || [];
        const pid = res?.persona_id || res?.data?.persona_id || res?.data?.persona?.id;
        setTraits(items);
        if (pid) setPersonaId(pid);
      } catch (e) {
        setError('Failed to load current traits');
      } finally {
        setLoadingTraits(false);
      }
    };
    load();
  }, []);

  // Real-time refresh: subscribe to persona.evolution events
  useEffect(() => {
    let mounted = true;
    let socketRef = null;
    (async () => {
      try {
        socketRef = await personaService.initSocket();
        if (!mounted) return;
        socketRef.on('persona.evolution', async (_payload) => {
          try {
            // Re-fetch current default persona traits
            const res = await personaService.getDefaultPersonaTraits();
            const items = res?.data?.traits || res?.traits || [];
            const pid = res?.persona_id || res?.data?.persona_id || res?.data?.persona?.id;
            setTraits(items);
            if (pid) setPersonaId(pid);
            // If current selected trait is present, refresh its history
            if (selectedTrait && pid) {
              setLastAction('loadHistory');
              const r = await personaService.getTraitHistory(pid, selectedTrait.name, { limit: 15 });
              const histRaw = r?.data?.history || r?.history || [];
              const hist = histRaw.map(h => ({
                ...h,
                delta: (typeof h.new_value === 'number' && typeof h.old_value === 'number')
                  ? (h.new_value - h.old_value)
                  : (typeof h.delta === 'number' ? h.delta : null),
                rationale: h.reasoning ?? h.rationale ?? '',
              }));
              setHistory(hist);
            }
          } catch (err) {
            // soft fail; UI will retry on next event/user action
            logger.debug('traits refresh failed after evolution event', err);
          }
        });
      } catch (e) {
        logger.debug('persona analytics socket init failed', e);
      }
    })();
    return () => {
      mounted = false;
      if (socketRef) socketRef.off('persona.evolution');
    };
  }, [selectedTrait]);

  const onSelectTrait = async (t) => {
    setSelectedTrait(t);
    setHistory([]);
    if (!t || !t.name) return;
    try {
      setLoadingHistory(true);
      setError(null);
      // Require personaId; skip if unavailable
      if (!personaId) {
        setLoadingHistory(false);
        return;
      }
      setLastAction('loadHistory');
      const r = await personaService.getTraitHistory(personaId, t.name, { limit: 15 });
      const histRaw = r?.data?.history || r?.history || [];
      const hist = histRaw.map(h => ({
        ...h,
        delta: (typeof h.new_value === 'number' && typeof h.old_value === 'number')
          ? (h.new_value - h.old_value)
          : (typeof h.delta === 'number' ? h.delta : null),
        rationale: h.reasoning ?? h.rationale ?? '',
      }));
      setHistory(hist);
    } catch (e) {
      setError('Failed to load trait history');
    } finally {
      setLoadingHistory(false);
    }
  };

  const radarTraits = useMemo(() => {
    return (traits || [])
      .filter(t => typeof t.value === 'number')
      .map(t => ({
        name: t.name,
        value: t.value,
        category: t.category,
      }));
  }, [traits]);

  // Build sparkline path from history values (min-max normalized)
  const Sparkline = ({ data, width = 160, height = 40, stroke = 'var(--color-accent)' }) => {
    if (!data || data.length < 2) return null;
    const values = data.map(d => (typeof d.value === 'number' ? d.value : (d.new_value ?? 0)));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const stepX = width / (values.length - 1);
    const pts = values.map((v, i) => {
      const x = i * stepX;
      const y = height - ((v - min) / range) * height;
      return `${x},${y}`;
    });
    const poly = pts.join(' ');
    const title = `Sparkline showing recent values for ${selectedTrait?.name || 'selected trait'}`;
    const desc = `Min ${min.toFixed(2)}, Max ${max.toFixed(2)}, ${values.length} points.`;
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="sparkline-title sparkline-desc">
        <title id="sparkline-title">{title}</title>
        <desc id="sparkline-desc">{desc}</desc>
        <polyline points={poly} fill="none" stroke={stroke} strokeWidth="2" />
      </svg>
    );
  };

  return (
    <div className="bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] rounded-lg p-4">
      <h3 className="text-lg font-semibold text-[var(--color-accent)] mb-3">Persona Analytics</h3>

      {error && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm" role="alert">
          <div className="flex items-center justify-between gap-3">
            <span>{error}</span>
            <button
              className="px-2 py-1 text-xs rounded border border-[var(--color-border)] hover:bg-[var(--color-bg-elev-2)] text-[var(--color-text-secondary)]"
              onClick={() => {
                if (lastAction === 'loadHistory' && selectedTrait) {
                  onSelectTrait(selectedTrait);
                } else {
                  // retry traits
                  (async () => {
                    try {
                      setLoadingTraits(true);
                      setError(null);
                      const res = await personaService.getDefaultPersonaTraits();
                      const items = res?.data?.traits || res?.traits || [];
                      const pid = res?.persona_id || res?.data?.persona_id || res?.data?.persona?.id;
                      setTraits(items);
                      if (pid) setPersonaId(pid);
                    } catch (e) {
                      setError('Failed to load current traits');
                    } finally {
                      setLoadingTraits(false);
                    }
                  })();
                }
              }}
            >Retry</button>
          </div>
        </div>
      )}

      {/* Top 6 Radar + Full Trait List Side-by-Side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="border border-[var(--color-border)] rounded p-3">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-[var(--color-text-secondary)]">Persona Trait Radar</h4>
            {loadingTraits && <span className="text-xs text-[var(--color-text-muted)]">Loading...</span>}
          </div>
          {radarTraits && radarTraits.length > 0 ? (
            <PersonaTraitRadar traits={radarTraits} />
          ) : (
            <div className="p-3 text-[var(--color-text-muted)] text-sm" role="status">No trait data yet</div>
          )}
        </div>
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-[var(--color-text-secondary)]">All Traits</h4>
            {loadingTraits && <span className="text-xs text-[var(--color-text-muted)]">Loading...</span>}
          </div>
          <div className="max-h-64 overflow-y-auto divide-y divide-[var(--color-border)] border border-[var(--color-border)] rounded" aria-live="polite" aria-busy={loadingTraits}>
            {loadingTraits ? (
              <div className="p-3 space-y-2">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-6 bg-[var(--color-bg-elev-2)] rounded animate-pulse" />
                ))}
              </div>
            ) : traits.length === 0 ? (
              <div className="p-3 text-[var(--color-text-muted)] text-sm" role="status">No traits available</div>
            ) : (
              traits.map((t, idx) => (
                <button
                  key={idx}
                  className={`w-full flex items-center justify-between p-3 text-left hover:bg-[var(--color-bg-elev-2)] transition-colors ${selectedTrait?.name === t.name ? 'bg-[var(--color-bg-elev-2)]' : ''}`}
                  onClick={() => onSelectTrait(t)}
                  aria-pressed={selectedTrait?.name === t.name}
                  aria-label={`Select trait ${formatTraitName(t.name)} with current value ${typeof t.value === 'number' ? t.value.toFixed(2) : '-'}`}
                >
                  <span className="text-[var(--color-text-primary)] text-sm truncate">{formatTraitName(t.name)}</span>
                  <span className="text-[var(--color-accent)] text-sm ml-3">{typeof t.value === 'number' ? t.value.toFixed(2) : '-'}</span>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Trait History */}
      <div className="mt-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium text-[var(--color-text-secondary)]">Trait History{selectedTrait?.name ? `: ${formatTraitName(selectedTrait.name)}` : ''}</h4>
          {loadingHistory && <span className="text-xs text-[var(--color-text-muted)]" role="status" aria-live="polite">Loading...</span>}
        </div>
        {/* Sparkline for selected trait */}
        {selectedTrait && history.length > 1 && !loadingHistory && (
          <div className="mb-2 px-2 py-1 bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded">
            <Sparkline data={history} />
          </div>
        )}
        <div className="max-h-48 overflow-y-auto border border-[var(--color-border)] rounded" aria-live="polite" aria-busy={loadingHistory}>
          {loadingHistory ? (
            <div className="p-3 space-y-3">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="space-y-1">
                  <div className="h-3 bg-[var(--color-bg-elev-2)] rounded animate-pulse" />
                  <div className="h-3 w-2/3 bg-[var(--color-bg-elev-2)] rounded animate-pulse" />
                </div>
              ))}
            </div>
          ) : (!selectedTrait || history.length === 0) ? (
            <div className="p-3 text-[var(--color-text-muted)] text-sm" role="status">
              {selectedTrait ? 'No history entries.' : 'Select a trait to view history.'}
            </div>
          ) : (
            <ul className="divide-y divide-[var(--color-border)]">
              {history.map((h, i) => (
                <li key={i} className="p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[var(--color-text-secondary)]">Δ {typeof h.delta === 'number' ? h.delta.toFixed(3) : h.delta}</span>
                    <span className="text-[var(--color-text-muted)]">{h.timestamp ? new Date(h.timestamp).toLocaleString() : ''}</span>
                  </div>
                  {h.rationale && (
                    <div className="mt-1 text-[var(--color-text-muted)] text-xs line-clamp-2">{h.rationale}</div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default PersonaAnalytics;
