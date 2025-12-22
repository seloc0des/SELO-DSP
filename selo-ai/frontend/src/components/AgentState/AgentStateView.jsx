import React, { useState, useEffect } from 'react';
import { getAllAgentState } from '../../services/agentStateService';
import { createLogger } from '../../utils/logger';

const logger = createLogger('AgentStateView');
import AffectiveStateCard from './AffectiveStateCard';
import GoalsList from './GoalsList';
import EpisodeTimeline from './EpisodeTimeline';
import MetaDirectivesList from './MetaDirectivesList';

const AgentStateView = ({ userId }) => {
  const [agentState, setAgentState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch agent state data
  const fetchAgentState = async () => {
    try {
      setError(null);
      const data = await getAllAgentState(null); // null = use default persona
      setAgentState(data);
      setLastUpdate(new Date());
    } catch (err) {
      logger.error('Failed to fetch agent state:', err);
      setError(err.message || 'Failed to load agent state');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    if (userId) {
      fetchAgentState();
    }
  }, [userId]);

  // Poll for updates every 5 seconds when tab is active
  useEffect(() => {
    if (!userId) return;

    const interval = setInterval(() => {
      fetchAgentState();
    }, 5000);

    return () => clearInterval(interval);
  }, [userId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--color-accent)] mx-auto mb-4"></div>
          <p className="text-[var(--color-text-muted)]">Loading agent state...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-4">
        <div className="flex items-start gap-3">
          <svg className="w-5 h-5 text-red-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-300 mb-1">Failed to load agent state</h3>
            <p className="text-sm text-red-200/80">{error}</p>
            <button 
              onClick={fetchAgentState}
              className="mt-3 text-xs text-red-300 hover:text-red-200 underline"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!agentState) {
    return (
      <div className="text-center py-12">
        <p className="text-[var(--color-text-muted)]">No agent state data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with last update time */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-[var(--color-text-primary)]">Agent State</h2>
          <p className="text-sm text-[var(--color-text-muted)] mt-1">
            Real-time emotional state, goals, and experiences
          </p>
        </div>
        <div className="text-right">
          {lastUpdate && (
            <p className="text-xs text-[var(--color-text-muted)]">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </p>
          )}
          <button 
            onClick={fetchAgentState}
            className="text-xs text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] mt-1"
          >
            Refresh now
          </button>
        </div>
      </div>

      {/* Affective State Card */}
      <AffectiveStateCard data={agentState.affective} />

      {/* Two-column layout for Goals and Directives */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GoalsList data={agentState.goals} />
        <MetaDirectivesList data={agentState.directives} />
      </div>

      {/* Episode Timeline (full width) */}
      <EpisodeTimeline data={agentState.episodes} />
    </div>
  );
};

export default AgentStateView;
