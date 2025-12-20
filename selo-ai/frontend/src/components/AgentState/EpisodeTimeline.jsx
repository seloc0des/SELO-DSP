import React from 'react';

const EpisodeTimeline = ({ data }) => {
  if (!data || !data.success || !data.data) {
    return (
      <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Autobiographical Episodes</h3>
        <p className="text-sm text-[var(--color-text-muted)]">No episodes data available</p>
      </div>
    );
  }

  const episodes = data.data.episodes || [];

  const getIntensityColor = (intensity) => {
    if (intensity >= 0.7) return 'bg-red-500/20 border-red-500/50 text-red-300';
    if (intensity >= 0.4) return 'bg-amber-500/20 border-amber-500/50 text-amber-300';
    return 'bg-blue-500/20 border-blue-500/50 text-blue-300';
  };

  const getEmotionEmoji = (emotion) => {
    const emojiMap = {
      'happy': '😊',
      'joyful': '😄',
      'excited': '🤩',
      'sad': '😢',
      'anxious': '😰',
      'worried': '😟',
      'calm': '😌',
      'curious': '🤔',
      'grateful': '🙏',
      'proud': '😌',
      'angry': '😠',
      'frustrated': '😤',
      'confused': '😕',
      'surprised': '😲',
      'hopeful': '🌟',
      'content': '😊',
      'peaceful': '☮️',
      'inspired': '✨',
    };
    return emojiMap[emotion?.toLowerCase()] || '💭';
  };

  return (
    <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <svg className="w-6 h-6 text-[var(--color-accent)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">Autobiographical Episodes</h3>
        </div>
        <span className="text-sm font-medium text-[var(--color-text-muted)]">
          {episodes.length} {episodes.length === 1 ? 'episode' : 'episodes'}
        </span>
      </div>

      {episodes.length === 0 ? (
        <div className="text-center py-12">
          <svg className="w-16 h-16 text-[var(--color-text-muted)] mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          <p className="text-sm text-[var(--color-text-muted)]">No episodes yet</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1">Significant moments are automatically captured</p>
        </div>
      ) : (
        <div className="space-y-4">
          {episodes.map((episode, index) => (
            <div 
              key={episode.id}
              className="relative bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg p-5 hover:border-[var(--color-accent)]/30 transition-colors"
            >
              {/* Timeline connector */}
              {index < episodes.length - 1 && (
                <div className="absolute left-8 top-full h-4 w-0.5 bg-[var(--color-border)]" />
              )}

              <div className="flex gap-4">
                {/* Emoji indicator */}
                <div className="text-3xl shrink-0 pt-1">
                  {getEmotionEmoji(episode.primary_emotion)}
                </div>

                <div className="flex-1 min-w-0">
                  {/* Title and timestamp */}
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <h4 className="text-base font-semibold text-[var(--color-text-primary)] leading-snug">
                      {episode.title || 'Untitled Episode'}
                    </h4>
                    {episode.created_at && (
                      <span className="text-xs text-[var(--color-text-muted)] shrink-0">
                        {new Date(episode.created_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>

                  {/* Narrative */}
                  {episode.narrative && (
                    <p className="text-sm text-[var(--color-text-secondary)] mb-3 leading-relaxed">
                      {episode.narrative}
                    </p>
                  )}

                  {/* Metadata row */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Emotional intensity */}
                    {typeof episode.emotional_intensity === 'number' && (
                      <span className={`text-xs px-2 py-1 rounded border ${getIntensityColor(episode.emotional_intensity)}`}>
                        Intensity: {Math.round(episode.emotional_intensity * 100)}%
                      </span>
                    )}

                    {/* Primary emotion */}
                    {episode.primary_emotion && (
                      <span className="text-xs px-2 py-1 bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded text-[var(--color-text-muted)]">
                        {episode.primary_emotion}
                      </span>
                    )}

                    {/* Trigger reason */}
                    {episode.trigger_reason && (
                      <span className="text-xs px-2 py-1 bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded text-[var(--color-text-muted)]">
                        {episode.trigger_reason.replace(/_/g, ' ')}
                      </span>
                    )}
                  </div>

                  {/* Key moments (if available) */}
                  {episode.key_moments && episode.key_moments.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
                      <p className="text-xs font-medium text-[var(--color-text-muted)] mb-2">Key Moments:</p>
                      <ul className="space-y-1">
                        {episode.key_moments.slice(0, 3).map((moment, idx) => (
                          <li key={idx} className="text-xs text-[var(--color-text-secondary)] flex items-start gap-2">
                            <span className="text-[var(--color-accent)] shrink-0">•</span>
                            <span className="flex-1">{moment}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default EpisodeTimeline;
