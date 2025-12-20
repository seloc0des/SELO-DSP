import React from 'react';

const AffectiveStateCard = ({ data }) => {
  if (!data || !data.success || !data.data) {
    return (
      <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Affective State</h3>
        <p className="text-sm text-[var(--color-text-muted)]">No affective state data available</p>
      </div>
    );
  }

  const state = data.data;
  
  // Helper to render progress bar
  const renderBar = (value, label, color = 'var(--color-accent)') => {
    const percentage = Math.round(value * 100);
    
    return (
      <div className="mb-4 last:mb-0">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-[var(--color-text-secondary)]">{label}</span>
          <span className="text-sm font-semibold text-[var(--color-text-primary)]">{percentage}%</span>
        </div>
        <div className="w-full bg-[var(--color-bg)] rounded-full h-2 overflow-hidden">
          <div 
            className="h-full rounded-full transition-all duration-500"
            style={{ 
              width: `${percentage}%`,
              backgroundColor: color
            }}
          />
        </div>
      </div>
    );
  };

  // Helper to get color based on value
  const getEnergyColor = (val) => {
    if (val < 0.3) return '#ef4444'; // red
    if (val < 0.6) return '#f59e0b'; // amber
    return '#10b981'; // green
  };

  const getStressColor = (val) => {
    if (val > 0.7) return '#ef4444'; // red (high stress)
    if (val > 0.4) return '#f59e0b'; // amber
    return '#10b981'; // green (low stress)
  };

  const getMoodColor = () => {
    const valence = state.mood_vector?.valence || 0;
    if (valence > 0.3) return '#10b981'; // green (positive)
    if (valence < -0.3) return '#ef4444'; // red (negative)
    return '#f59e0b'; // amber (neutral)
  };

  return (
    <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <svg className="w-6 h-6 text-[var(--color-accent)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
        </svg>
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">Affective State</h3>
      </div>

      <div className="space-y-1">
        {/* Energy */}
        {typeof state.energy === 'number' && renderBar(
          state.energy,
          'Energy',
          getEnergyColor(state.energy)
        )}

        {/* Stress */}
        {typeof state.stress === 'number' && renderBar(
          state.stress,
          'Stress',
          getStressColor(state.stress)
        )}

        {/* Confidence */}
        {typeof state.confidence === 'number' && renderBar(
          state.confidence,
          'Confidence',
          'var(--color-accent)'
        )}

        {/* Mood Vector */}
        {state.mood_vector && (
          <div className="mt-6 pt-6 border-t border-[var(--color-border)]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-[var(--color-text-secondary)]">Mood</span>
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: getMoodColor() }}
                title={`Valence: ${state.mood_vector.valence?.toFixed(2) || 0}, Arousal: ${state.mood_vector.arousal?.toFixed(2) || 0}`}
              />
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-[var(--color-text-muted)]">Valence:</span>
                <span className="ml-2 font-medium text-[var(--color-text-primary)]">
                  {state.mood_vector.valence?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div>
                <span className="text-[var(--color-text-muted)]">Arousal:</span>
                <span className="ml-2 font-medium text-[var(--color-text-primary)]">
                  {state.mood_vector.arousal?.toFixed(2) || '0.00'}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Last Update */}
        {state.updated_at && (
          <div className="mt-4 pt-4 border-t border-[var(--color-border)]">
            <p className="text-xs text-[var(--color-text-muted)]">
              Updated: {new Date(state.updated_at).toLocaleString()}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AffectiveStateCard;
