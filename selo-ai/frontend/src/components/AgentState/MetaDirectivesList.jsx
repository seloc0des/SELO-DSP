import React from 'react';

const MetaDirectivesList = ({ data }) => {
  if (!data || !data.success || !data.data) {
    return (
      <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Meta-Directives</h3>
        <p className="text-sm text-[var(--color-text-muted)]">No directives data available</p>
      </div>
    );
  }

  const directives = data.data.meta_directives || [];

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'in_progress':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      case 'completed':
        return 'bg-green-500/10 text-green-400 border-green-500/30';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/30';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'pending':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
      case 'in_progress':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        );
      case 'completed':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <svg className="w-6 h-6 text-[var(--color-accent)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">Meta-Directives</h3>
        </div>
        <span className="text-sm font-medium text-[var(--color-text-muted)]">
          {directives.length} active
        </span>
      </div>

      {directives.length === 0 ? (
        <div className="text-center py-8">
          <svg className="w-12 h-12 text-[var(--color-text-muted)] mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <p className="text-sm text-[var(--color-text-muted)]">No meta-directives yet</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1">Self-assigned tasks from reflections</p>
        </div>
      ) : (
        <div className="space-y-3">
          {directives.map((directive) => (
            <div 
              key={directive.id}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg p-4 hover:border-[var(--color-accent)]/30 transition-colors"
            >
              <div className="flex items-start gap-3 mb-3">
                <div className={`flex items-center gap-1.5 px-2 py-1 rounded border text-xs font-medium ${getStatusColor(directive.status)}`}>
                  {getStatusIcon(directive.status)}
                  <span>{directive.status?.replace('_', ' ')}</span>
                </div>
              </div>

              <p className="text-sm text-[var(--color-text-primary)] mb-2 leading-relaxed">
                {directive.directive_text}
              </p>

              {directive.rationale && (
                <p className="text-xs text-[var(--color-text-muted)] mb-3 italic">
                  {directive.rationale}
                </p>
              )}

              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  {directive.source_type && (
                    <span className="px-2 py-0.5 bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded text-[var(--color-text-muted)]">
                      {directive.source_type}
                    </span>
                  )}
                </div>
                {directive.created_at && (
                  <span className="text-[var(--color-text-muted)]">
                    {new Date(directive.created_at).toLocaleDateString()}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MetaDirectivesList;
