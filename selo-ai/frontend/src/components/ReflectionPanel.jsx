import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { reflectionService } from '../services/reflectionService';
import ErrorBoundary from './ErrorBoundary';
import ReflectionCard from './Reflection/ReflectionCard';
import { formatRelativeTime } from '../utils/dateFormatter';
import { reflectionLogger as logger } from '../utils/logger';

const ReflectionPanel = ({ sessionId, messages }) => {
  // Load reflections from localStorage on mount (if available for this session)
  const getStoredReflections = useCallback(() => {
    try {
      const key = `reflections_${sessionId}`;
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      logger.warn('Failed to load reflections from localStorage:', e);
      return [];
    }
  }, [sessionId]);

  const [reflections, setReflections] = useState(getStoredReflections);
  // Only show loading spinner if we have no cached data
  const [loading, setLoading] = useState(() => getStoredReflections().length === 0);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [isPolling, setIsPolling] = useState(false);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [reflectingCount, setReflectingCount] = useState(0);

  // Persist reflections to localStorage whenever they change
  useEffect(() => {
    if (reflections.length > 0) {
      const saveToLocalStorage = () => {
        const key = `reflections_${sessionId}`;
        try {
          localStorage.setItem(key, JSON.stringify(reflections));
          // Store timestamp for this session
          localStorage.setItem(`${key}_timestamp`, new Date().toISOString());
          return true;
        } catch (e) {
          if (e.name === 'QuotaExceededError') {
            // Proactively clean up old data when quota exceeded
            try {
              const allKeys = Object.keys(localStorage);
              const reflectionKeys = allKeys
                .filter(k => k.startsWith('reflections_') && k !== key)
                .slice(0, Math.floor(allKeys.length / 2));
              
              // Remove old reflection data
              reflectionKeys.forEach(k => {
                localStorage.removeItem(k);
                localStorage.removeItem(`${k}_timestamp`);
              });
              
              // Retry save after cleanup
              localStorage.setItem(key, JSON.stringify(reflections));
              localStorage.setItem(`${key}_timestamp`, new Date().toISOString());
              return true;
            } catch (retryErr) {
              logger.error('Failed to save reflections even after cleanup:', retryErr);
              return false;
            }
          } else {
            logger.warn('Failed to save reflections to localStorage:', e);
            return false;
          }
        }
      };
      
      const saved = saveToLocalStorage();
      
      // Periodic cleanup: Remove old reflection data (keep only last 3 sessions)
      if (saved && Math.random() < 0.1) { // 10% chance to run cleanup
        try {
          const allKeys = Object.keys(localStorage);
          const reflectionKeys = allKeys
            .filter(k => k.startsWith('reflections_'))
            .map(k => ({
              key: k.replace('_timestamp', ''),
              timestamp: localStorage.getItem(`${k}_timestamp`) || '0'
            }))
            .filter((item, index, self) => 
              index === self.findIndex(t => t.key === item.key)
            )
            .sort((a, b) => b.timestamp.localeCompare(a.timestamp));
          
          // Keep only the 3 most recent sessions
          if (reflectionKeys.length > 3) {
            reflectionKeys.slice(3).forEach(({ key }) => {
              localStorage.removeItem(key);
              localStorage.removeItem(`${key}_timestamp`);
            });
          }
        } catch (cleanupErr) {
          logger.warn('Periodic cleanup failed:', cleanupErr);
        }
      }
    }
  }, [reflections, sessionId]);

  const ITEMS_PER_PAGE = 20;
  const MAX_RETRIES = 3;

  // Memoized filtered reflections for current page
  // Show all user-facing reflections (message, daily), exclude only internal seeds
  const paginatedReflections = useMemo(() => {
    const userFacingReflections = reflections.filter(r => {
      const type = r.reflection_type || r.type;
      const metadata = r.metadata || {};
      // Exclude internal seeds and scheduled system tasks, but include message and daily reflections
      if (metadata.seed || metadata.system_task) return false;
      return type === 'message' || type === 'daily';
    });
    return userFacingReflections.slice(0, page * ITEMS_PER_PAGE);
  }, [reflections, page]);

  // Use ref to store initialization function to avoid dependency issues
  const initializeReflectionsRef = useRef();
  initializeReflectionsRef.current = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch initial reflections and filter to user-facing types
      const initialReflections = await reflectionService.getReflections(sessionId);
      const userFacingReflections = (initialReflections || []).filter(r => {
        const type = r.reflection_type || r.type;
        const metadata = r.metadata || {};
        if (metadata.seed || metadata.system_task) return false;
        return type === 'message' || type === 'daily';
      });
      setReflections(userFacingReflections);
      setHasMore(userFacingReflections.length >= ITEMS_PER_PAGE);
      setLoading(false);
      setRetryCount(0); // Reset retry count on success

      // Set up real-time updates
      const unsubscribe = reflectionService.onReflectionGenerated((evt) => {
        // evt may be a wrapped event with status 'generating' | 'complete'
        const base = evt || {};
        if (base.status === 'generating') {
          setReflectingCount(c => c + 1);
        }
        const payload = base?.data || base;
        if (!payload) return;
        if (!payload.user_profile_id || payload.user_profile_id === sessionId) {
          if (base.status === 'complete') {
            setReflectingCount(c => Math.max(0, c - 1));
            // Add user-facing reflections, skip scheduled tasks and internal seeds
            const type = payload.reflection_type || payload.type;
            const metadata = payload.metadata || {};
            if (!metadata.seed && !metadata.system_task && (type === 'message' || type === 'daily')) {
              setReflections(prev => {
                const newId = payload.reflection_id || base.reflectionId;
                const exists = prev.some(r => (r.reflection_id || r.reflectionId) === newId);
                return exists ? prev : [payload, ...prev];
              });
            }
          }
        }
      }, sessionId);

      // Return cleanup for the caller if needed
      return () => {
        try { unsubscribe && unsubscribe(); } catch (e) { /* noop */ }
      };
    } catch (err) {
      logger.error('Failed to initialize reflections:', err);
      setError(err.message || 'Failed to load reflections');
      setLoading(false);
    }
  };

  const handleRetry = useCallback(async () => {
    if (retryCount < MAX_RETRIES) {
      setRetryCount(prev => prev + 1);
      setError(null);
      setLoading(true);
      await initializeReflectionsRef.current();
    }
  }, [retryCount]);

  // Subscribe to connection status once (mount/unmount)
  useEffect(() => {
    const unsubscribeConn = reflectionService.subscribeToConnectionStatus((status) => {
      setConnectionStatus(status);
      setIsPolling(status === 'polling');
    });
    return () => {
      try { unsubscribeConn && unsubscribeConn(); } catch (e) { /* noop */ }
    };
  }, []);

  // Initialize reflections when session changes (do not depend on connectionStatus to avoid teardown loops)
  useEffect(() => {
    let mounted = true;
    let unsubscribeRefl = null;

    const setup = async () => {
      if (!sessionId) return;
      unsubscribeRefl = await initializeReflectionsRef.current();
    };

    setup();

    return () => {
      try { unsubscribeRefl && unsubscribeRefl(); } catch (e) { /* noop */ }
      // Intentionally do NOT disconnect the socket here; let the service manage its lifecycle
    };
  }, [sessionId]);

  // Manage polling separately based on connection status without tearing down the socket
  useEffect(() => {
    let pollInterval = null;
    if (!sessionId) return undefined;
    if (connectionStatus === 'offline' || connectionStatus === 'polling') {
      pollInterval = setInterval(async () => {
        try {
          const latestReflections = await reflectionService.getReflections(sessionId);
          // Filter to user-facing reflections only
          const userFacingReflections = (latestReflections || []).filter(r => {
            const type = r.reflection_type || r.type;
            const metadata = r.metadata || {};
            if (metadata.seed || metadata.system_task) return false;
            return type === 'message' || type === 'daily';
          });
          setReflections(userFacingReflections);
        } catch (err) {
          logger.error('Polling failed:', err);
        }
      }, 5000);
    }

    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [connectionStatus, sessionId]);

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-400';
      case 'connecting': return 'text-yellow-400';
      case 'offline': return 'text-red-400';
      case 'polling': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  const getConnectionStatusText = () => {
    if (isPolling) return 'Polling';
    switch (connectionStatus) {
      case 'connected': return 'Live';
      case 'connecting': return 'Connecting';
      case 'offline': return 'Offline';
      default: return 'Unknown';
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown time';
    try {
      const d = new Date(timestamp);
      if (isNaN(d.getTime())) return 'Invalid time';
      return d.toLocaleString();
    } catch {
      return 'Invalid time';
    }
  };

  if (loading) {
    return (
      <div className="w-full max-w-md lg:max-w-lg bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-[var(--color-accent)]">SELO's Inner Reflections</h3>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-yellow-400">Loading...</span>
          </div>
        </div>
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-[var(--color-bg-elev-2)] rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-[var(--color-bg-elev-2)] rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="w-full max-w-md lg:max-w-lg bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] rounded-lg p-3 sm:p-4 h-80 sm:h-96 flex flex-col">
        {/* Header with connection status */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-[var(--color-accent)]">SELO's Inner Reflections</h3>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-400' : 
              connectionStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' : 
              connectionStatus === 'polling' ? 'bg-blue-400 animate-pulse' :
              'bg-red-400'
            }`}></div>
            <span className={`text-xs ${getConnectionStatusColor()}`}>{getConnectionStatusText()}</span>
          </div>
        </div>

        {/* Error state */}
        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
            <div className="flex items-center justify-between">
              <span>Error: {error}</span>
              {retryCount < MAX_RETRIES && (
                <button 
                  onClick={handleRetry}
                  className="ml-2 px-2 py-1 bg-red-500/20 hover:bg-red-500/30 rounded text-xs transition-colors"
                >
                  Retry ({MAX_RETRIES - retryCount} left)
                </button>
              )}
            </div>
          </div>
        )}

        {/* Reflections list */}
        <div className="flex-1 overflow-y-auto space-y-3">
          {reflectingCount > 0 && (
            <div className="p-3 bg-[var(--color-bg-elev-2)] rounded border border-[var(--color-border)]">
              <div className="flex items-center space-x-2 text-[var(--color-accent)]">
                <div className="w-2 h-2 bg-[var(--color-accent)] rounded-full animate-pulse"></div>
                <span className="text-sm">Reflecting…</span>
              </div>
            </div>
          )}
          {paginatedReflections.length === 0 ? (
            <div className="text-center text-[var(--color-text-muted)] py-8">
              <div className="text-2xl mb-2">🤔</div>
              <p>No reflections yet...</p>
              <p className="text-xs mt-1">SELO will reflect on your conversation</p>
            </div>
          ) : (
            paginatedReflections.map((reflection, index) => {
              const createdAt = reflection.created_at || reflection.timestamp;
              const timeAgo = formatRelativeTime(createdAt);
              
              return (
                <ErrorBoundary key={reflection.id || reflection.reflection_id || index}>
                  <ReflectionCard
                    reflection={reflection}
                    timeAgo={timeAgo}
                  />
                </ErrorBoundary>
              );
            })
          )}
        </div>

        {/* Load more button */}
        {hasMore && paginatedReflections.length > 0 && (
          <div className="mt-4 text-center">
            <button
              onClick={() => setPage(prev => prev + 1)}
              className="px-4 py-2 text-[var(--color-accent)] border border-[var(--color-accent)] hover:bg-[var(--color-accent)] hover:text-black rounded text-sm transition-colors"
            >
              Load More
            </button>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default ReflectionPanel;
