import React from 'react';

const GoalsList = ({ data }) => {
  if (!data || !data.success || !data.data) {
    return (
      <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Active Goals</h3>
        <p className="text-sm text-[var(--color-text-muted)]">No goals data available</p>
      </div>
    );
  }

  const goals = data.data.goals || [];

  const getPriorityColor = (priority) => {
    if (priority >= 0.8) return 'text-red-400';
    if (priority >= 0.5) return 'text-amber-400';
    return 'text-blue-400';
  };

  const getPriorityLabel = (priority) => {
    if (priority >= 0.8) return 'High';
    if (priority >= 0.5) return 'Medium';
    return 'Low';
  };

  return (
    <div className="bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <svg className="w-6 h-6 text-[var(--color-accent)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">Active Goals</h3>
        </div>
        <span className="text-sm font-medium text-[var(--color-text-muted)]">
          {goals.length} {goals.length === 1 ? 'goal' : 'goals'}
        </span>
      </div>

      {goals.length === 0 ? (
        <div className="text-center py-8">
          <svg className="w-12 h-12 text-[var(--color-text-muted)] mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          <p className="text-sm text-[var(--color-text-muted)]">No active goals yet</p>
          <p className="text-xs text-[var(--color-text-muted)] mt-1">Goals are created from reflection action items</p>
        </div>
      ) : (
        <div className="space-y-3">
          {goals.map((goal) => (
            <div 
              key={goal.id}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded-lg p-4 hover:border-[var(--color-accent)]/30 transition-colors"
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <h4 className="text-sm font-medium text-[var(--color-text-primary)] flex-1 leading-snug">
                  {goal.title}
                </h4>
                <span className={`text-xs font-medium ${getPriorityColor(goal.priority)} shrink-0`}>
                  {getPriorityLabel(goal.priority)}
                </span>
              </div>

              {goal.description && (
                <p className="text-xs text-[var(--color-text-muted)] mb-3 line-clamp-2">
                  {goal.description}
                </p>
              )}

              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  {goal.origin && (
                    <span className="px-2 py-0.5 bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] rounded text-[var(--color-text-muted)]">
                      {goal.origin}
                    </span>
                  )}
                  {goal.status && (
                    <span className={`px-2 py-0.5 rounded ${
                      goal.status === 'active' ? 'bg-green-500/10 text-green-400' :
                      goal.status === 'paused' ? 'bg-amber-500/10 text-amber-400' :
                      'bg-gray-500/10 text-gray-400'
                    }`}>
                      {goal.status}
                    </span>
                  )}
                </div>
                {goal.created_at && (
                  <span className="text-[var(--color-text-muted)]">
                    {new Date(goal.created_at).toLocaleDateString()}
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

export default GoalsList;
