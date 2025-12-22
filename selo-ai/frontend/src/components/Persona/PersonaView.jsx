import React, { useState, useEffect, useCallback, useRef } from 'react';
import PersonaCard from './PersonaCard';
import PersonaEvolution from './PersonaEvolution';
import PersonaAnalytics from './PersonaAnalytics';
import personaService from '../../services/personaService';
import { personaLogger as logger } from '../../utils/logger';
import ErrorBoundary from '../ErrorBoundary';

/**
 * Main component for displaying persona information
 * Shows persona cards and evolution history
 */
const PersonaView = ({ userId }) => {
  const [personas, setPersonas] = useState([]);
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [evolutionHistory, setEvolutionHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Polling removed for single-persona setup; real-time updates come via sockets

  // Track if we've already set initial persona to prevent loops
  const hasSetInitialPersona = useRef(false);

  // Define fetch functions with useCallback to prevent infinite loops
  // Removed selectedPersona from dependencies to prevent circular updates
  const fetchPersonas = useCallback(async () => {
    try {
      setLoading(true);
      const { personas } = await personaService.getPersonas(userId);
      setPersonas(personas);
      
      // Select first persona by default if none selected (only once)
      // Use ref check instead of state dependency to avoid circular updates
      if (personas.length > 0 && !hasSetInitialPersona.current) {
        setSelectedPersona(personas[0]);
        hasSetInitialPersona.current = true;
      }
      
      setLoading(false);
    } catch (err) {
      logger.error('Error fetching personas:', err);
      setError('Failed to load personas');
      setLoading(false);
    }
  }, [userId]);

  const fetchEvolutionHistory = useCallback(async (personaId) => {
    try {
      const res = await personaService.getPersonaEvolutionHistory(userId, personaId);
      const history = res?.history || [];
      setEvolutionHistory(history);
    } catch (err) {
      logger.error('Error fetching evolution history:', err);
      setError('Failed to load evolution history');
      setEvolutionHistory([]);
    }
  }, [userId]);

  // Initialize socket connection (persona events only)
  useEffect(() => {
    let isMounted = true;
    let socketRef = null;

    (async () => {
      try {
        const s = await personaService.initSocket();
        if (!isMounted) return;
        socketRef = s;

        // Listen for persona evolution events
        // Use functional updates to avoid stale closure on selectedPersona
        socketRef.on('persona.evolution', (data) => {
          logger.debug('Received persona.evolution event:', data);
          if (data.user_id === userId) {
            // Refresh persona data when evolution occurs
            fetchPersonas();
            // Use functional state update to get current selectedPersona
            setSelectedPersona(current => {
              if (current && data.persona_id === current.id) {
                fetchEvolutionHistory(current.id);
              }
              return current;
            });
          }
        });

        // Listen for initial bootstrap summary to populate UI immediately after bootstrap
        socketRef.on('persona.bootstrap_summary', (data) => {
          logger.debug('Received persona.bootstrap_summary event:', data);
          if (data.user_id === userId) {
            fetchPersonas();
            // Use functional state update to get current selectedPersona
            setSelectedPersona(current => {
              if (current && data.persona_id === current.id) {
                fetchEvolutionHistory(current.id);
              }
              return current;
            });
          }
        });

      } catch (e) {
        logger.error('Failed to initialize persona socket:', e);
      }
    })();

    return () => {
      isMounted = false;
      // Clean up event listeners
      if (socketRef) {
        socketRef.off('persona.evolution');
        socketRef.off('persona.bootstrap_summary');
      }
    };
  }, [userId, fetchPersonas, fetchEvolutionHistory]);

  // Fetch evolution history when selected persona changes
  useEffect(() => {
    if (selectedPersona) {
      fetchEvolutionHistory(selectedPersona.id);
    }
  }, [selectedPersona, fetchEvolutionHistory]);

  // Single effect to handle initial persona loading (avoid race conditions)
  useEffect(() => {
    if (userId) {
      const initializePersonas = async () => {
        try {
          setLoading(true);
          // fetchPersonas already ensures default persona exists and fetches it
          // No need for separate ensureDefaultPersona call (removes duplicate API call)
          await fetchPersonas();
        } catch (err) {
          logger.error('Failed to initialize personas:', err);
          setError('Failed to initialize personas');
          setLoading(false);
        }
      };
      
      initializePersonas();
    }
  }, [userId, fetchPersonas]);

  return (
    <ErrorBoundary>
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-[var(--color-accent)]">
            <span className="text-sm font-normal text-gray-400 ml-2">
              (Autonomous Evolution)
            </span>
          </h2>
          <button
            onClick={async () => {
              await fetchPersonas();
              if (selectedPersona) {
                await fetchEvolutionHistory(selectedPersona.id);
              }
            }}
            className="text-sm px-3 py-1 rounded-md bg-[var(--color-bg-elev-2)] hover:bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] text-[var(--color-accent)]"
            title="Refresh persona data"
          >
            Refresh
          </button>
        </div>
        
        {error && (
          <div className="bg-red-900/30 border border-red-700 p-3 rounded-md mb-4 text-red-200">
            {error}
          </div>
        )}
        
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-pulse text-[var(--color-accent)]">Loading personas...</div>
          </div>
        ) : (
          <>
            {personas.length === 0 ? (
              <div className="bg-gray-800/50 p-6 rounded-md text-center">
                <p>No personas found. Creating default persona...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1">
                  <div className="bg-gray-900/50 rounded-lg p-4 h-full">
                    <h3 className="text-xl font-medium text-[var(--color-accent)] mb-1">SELO's Current State</h3>
                    <div className="text-2xl font-semibold text-[var(--color-text-primary)] mb-3">
                      {selectedPersona?.name || selectedPersona?.data?.name || 'SELO'}
                    </div>
                    <div className="space-y-4">
                      {personas.map(persona => (
                        <ErrorBoundary key={persona.id}>
                          <PersonaCard 
                            persona={persona}
                            isSelected={selectedPersona && selectedPersona.id === persona.id}
                            onClick={() => setSelectedPersona(persona)}
                          />
                        </ErrorBoundary>
                      ))}
                    </div>
                  </div>
                </div>
                
                <div className="lg:col-span-2 flex flex-col gap-6">
                  {selectedPersona ? (
                    <>
                      <ErrorBoundary>
                        <PersonaEvolution 
                          persona={selectedPersona} 
                          evolutionHistory={evolutionHistory}
                          userId={userId}
                        />
                      </ErrorBoundary>
                      <ErrorBoundary>
                        <PersonaAnalytics />
                      </ErrorBoundary>
                    </>
                  ) : (
                    <div className="bg-gray-800/50 p-6 rounded-md text-center">
                      <p>Select a persona to view its evolution history</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default PersonaView;
