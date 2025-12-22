import React, { useState, useEffect } from 'react';
import personaService from '../../services/personaService';
import { personaLogger as logger } from '../../utils/logger';
import { sanitizeContentText, parseJsonFromString, toArray } from '../Reflection/ReflectionCard';

/**
 * Component for displaying persona evolution history
 * Shows timeline of changes and a presentation block (intro or latest summary)
 */
const PersonaEvolution = ({ persona, evolutionHistory, userId }) => {
  const [presentation, setPresentation] = useState('');
  const [presentationType, setPresentationType] = useState(''); // 'first_intro' | 'last_session_summary' | ''
  const mantra = (persona && persona.mantra) ? persona.mantra : '';
  const [loading, setLoading] = useState(true);
  
  // Normalize evolution changes to an array of strings for robust rendering
  const normalizeChanges = (changes, sourceType) => {
    try {
      if (!changes) return [];
      if (Array.isArray(changes)) return changes.filter(Boolean).map(String);
      if (typeof changes === 'string') return [changes];
      if (typeof changes === 'object') {
        // Special handling for bootstrap evolutions
        if (sourceType === 'bootstrap') {
          const out = [];
          
          // Extract traits if present
          if (changes.traits && Array.isArray(changes.traits.traits)) {
            const traitCount = changes.traits.traits.length;
            out.push(`Generated ${traitCount} initial traits`);
          } else if (changes.traits && typeof changes.traits === 'object') {
            out.push('Initial traits established');
          }
          
          // Extract seed description
          if (changes.seed && changes.seed.description) {
            out.push(`Identity: ${changes.seed.description}`);
          }
          
          // Extract values
          if (changes.seed && changes.seed.values) {
            const values = changes.seed.values;
            if (values.core && Array.isArray(values.core)) {
              out.push(`Core values: ${values.core.join(', ')}`);
            }
            if (values.principles && Array.isArray(values.principles)) {
              out.push(`Principles: ${values.principles.join(', ')}`);
            }
          }
          
          // Extract knowledge domains
          if (changes.seed && changes.seed.knowledge_domains && Array.isArray(changes.seed.knowledge_domains)) {
            out.push(`Knowledge domains: ${changes.seed.knowledge_domains.join(', ')}`);
          }
          
          // Extract communication style
          if (changes.seed && changes.seed.communication_style) {
            const style = changes.seed.communication_style;
            if (style.tone) {
              out.push(`Communication tone: ${style.tone}`);
            }
          }
          
          return out.length > 0 ? out : ['Initial persona bootstrap completed'];
        }
        
        // Regular evolution changes
        const out = [];
        for (const [k, v] of Object.entries(changes)) {
          if (v === null || v === undefined) continue;
          
          // Special handling for trait changes - make them user-friendly
          if (k === 'traits' && Array.isArray(v)) {
            v.forEach(trait => {
              const name = trait.name || 'Unknown trait';
              const delta = trait.delta || 0;
              const reason = trait.reason || '';
              const direction = delta > 0 ? 'increased' : 'decreased';
              const percentage = Math.abs(Math.round(delta * 100));
              
              let line = `${name.charAt(0).toUpperCase() + name.slice(1)} ${direction} by ${percentage}%`;
              if (reason) {
                line += ` - ${reason}`;
              }
              out.push(line);
            });
          } else if (typeof v === 'object') {
            out.push(`${k}: ${JSON.stringify(v)}`);
          } else {
            out.push(`${k}: ${String(v)}`);
          }
        }
        return out;
      }
      return [String(changes)];
    } catch {
      return [];
    }
  };
  
  // Fetch presentation content when persona changes
  useEffect(() => {
    if (persona && persona.id && userId) {
      setLoading(true);
      personaService.getPersonaPresentation(userId, persona.id)
        .then(response => {
          const data = response && response.data ? response.data : {};
          const firstIntro = data.first_intro || '';
          const lastSummary = data.last_session_summary || '';
          const firstIntroUsed = !!data.first_intro_used;
          // Logic: if first intro has not been used yet and exists, prefer it; otherwise show last summary
          if (!firstIntroUsed && firstIntro) {
            setPresentation(firstIntro);
            setPresentationType('first_intro');
          } else {
            setPresentation(lastSummary || firstIntro || '');
            setPresentationType(lastSummary ? 'last_session_summary' : (firstIntro ? 'first_intro' : ''));
          }
          setLoading(false);
        })
        .catch(error => {
          logger.error('Failed to fetch persona presentation:', error);
          setLoading(false);
        });
    } else if (!userId) {
      // Handle missing userId case - don't try to fetch
      setLoading(false);
      setPresentation('');
      setPresentationType('');
    }
  }, [persona, userId]);
  
  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  return (
    <div className="bg-[var(--color-bg-elev-1)] rounded-lg p-4 h-full border border-[var(--color-border)]">
      <h3 className="text-xl font-medium text-[var(--color-accent)] mb-3">
        SELO's Evolution & Narrative
      </h3>
      
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-lg text-[var(--color-accent)] font-medium">
            Current Mantra
          </h4>
          <span className="text-xs px-2 py-1 rounded bg-gray-700/70 text-gray-300">
            v{persona.version || '1.0'}
          </span>
        </div>
        
        {loading ? (
          <div className="h-20 animate-pulse bg-gray-800/50 rounded p-3"></div>
        ) : (
          <div className="bg-gray-800/50 rounded p-3 text-sm max-h-60 overflow-y-auto whitespace-pre-wrap">
            {mantra}
          </div>
        )}
      </div>
      
      <div>
        <h4 className="text-lg text-[var(--color-accent)] font-medium mb-2">Evolution Timeline</h4>
        
        {evolutionHistory && evolutionHistory.length > 0 ? (
          <div className="relative pl-6 border-l border-[var(--color-border)] space-y-6">
            {evolutionHistory.map((entry, index) => {
              const normalizedChanges = normalizeChanges(entry.changes, entry.source_type);
              const evidence = entry.evidence || {};
              const rawReflectionContent = evidence.reflection_content;
              const parsedReflection = typeof rawReflectionContent === 'string'
                ? parseJsonFromString(rawReflectionContent)
                : (rawReflectionContent && typeof rawReflectionContent === 'object' ? rawReflectionContent : null);

              const reflectionContentText = rawReflectionContent
                ? sanitizeContentText(
                    typeof rawReflectionContent === 'string'
                      ? rawReflectionContent
                      : (parsedReflection?.content ?? '')
                  )
                : '';

              const reflectionThemes = (evidence.reflection_themes && evidence.reflection_themes.length > 0)
                ? evidence.reflection_themes
                : toArray(parsedReflection?.themes);

              const reflectionInsights = toArray(parsedReflection?.insights);
              const reflectionActions = toArray(parsedReflection?.actions);
              const reflectionEmotion = parsedReflection && typeof parsedReflection === 'object' && parsedReflection?.emotional_state && typeof parsedReflection.emotional_state === 'object'
                ? parsedReflection.emotional_state
                : null;

              const intensityDisplay = reflectionEmotion && typeof reflectionEmotion.intensity === 'number'
                ? (reflectionEmotion.intensity >= 0 && reflectionEmotion.intensity <= 1
                    ? `${Math.round(reflectionEmotion.intensity * 100)}%`
                    : reflectionEmotion.intensity.toFixed(2))
                : null;

              return (
              <div key={index} className="relative">
                {/* Timeline node */}
                <div 
                  className="absolute -left-[21px] w-4 h-4 rounded-full bg-[var(--color-accent)]"
                  style={{ top: '0px' }}
                ></div>
                
                {/* Evolution entry */}
                <div className="bg-[var(--color-bg-elev-2)] rounded p-3 pb-4 border border-[var(--color-border)]">
                  <div className="flex justify-between mb-1">
                    <span className="text-xs text-gray-400">
                      {formatDate(entry.timestamp)}
                    </span>
                    <div className="flex gap-2">
                      <span 
                        className={`text-xs px-2 rounded ${
                          entry.confidence >= 0.7 
                            ? 'bg-green-900/50 text-green-300' 
                            : entry.confidence >= 0.4 
                              ? 'bg-yellow-900/50 text-yellow-300'
                              : 'bg-red-900/50 text-red-300'
                        }`}
                      >
                        Confidence: {Math.round(entry.confidence * 100)}%
                      </span>
                      {entry.impact_score !== undefined && (
                        <span className={`text-xs px-2 rounded ${
                          entry.impact_score >= 0.7 
                            ? 'bg-purple-900/50 text-purple-300' 
                            : entry.impact_score >= 0.4 
                              ? 'bg-blue-900/50 text-blue-300'
                              : 'bg-gray-900/50 text-gray-300'
                        }`}>
                          Impact: {entry.impact_score >= 0.7 ? 'High' : entry.impact_score >= 0.4 ? 'Medium' : 'Low'}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <h5 className="text-sm font-medium text-white mb-1">
                    {(() => {
                      const sourceTypeLabels = {
                        'reflection': 'Reflection-Driven Evolution',
                        'learning': 'Learning-Based Growth',
                        'scheduled': 'Scheduled Reassessment',
                        'user_feedback': 'User Feedback Integration',
                        'bootstrap': 'Initial Bootstrap'
                      };
                      return sourceTypeLabels[entry.source_type] || 'System Update';
                    })()}
                  </h5>
                  
                  <p className="text-sm text-gray-300 mb-2">
                    {entry.source_type === 'bootstrap' 
                      ? 'Initial identity bootstrap - persona emerged with core traits, values, and communication style'
                      : (entry.reasoning || 'No description provided')
                    }
                  </p>
                  
                  <div className="mt-2 text-xs">
                    <div className="flex items-center gap-1 text-[var(--color-accent)] mb-1">
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M20 14.66V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5.34"></path>
                        <polygon points="18 2 22 6 12 16 8 16 8 12 18 2"></polygon>
                      </svg>
                      <span>Changes</span>
                    </div>
                    <ul className="list-disc list-inside pl-2 text-gray-400">
                      {normalizedChanges.map((change, idx) => (
                        <li key={idx}>{change}</li>
                      ))}
                      {normalizedChanges.length === 0 && (
                        <li className="text-gray-500 italic">No specific changes recorded</li>
                      )}
                    </ul>
                  </div>
                  
                  {(entry.learning_id || (entry.evidence && entry.evidence.reflection_id)) && (
                    <div className="mt-2 pt-2 border-t border-gray-700/50 text-xs">
                      {entry.learning_id && (
                        <div className="text-gray-500">
                          Learning ID: {entry.learning_id.substring(0, 8)}...
                        </div>
                      )}
                      {entry.evidence && entry.evidence.reflection_id && (
                        <div>
                          <div className="mb-1">
                            <span className="text-gray-500">Triggered by: </span>
                            <span className="text-[var(--color-accent)] font-mono text-xs">
                              Reflection {entry.evidence.reflection_id.substring(0, 8)}...
                            </span>
                          </div>
                          
                          {/* Display reflection themes if available */}
                          {reflectionThemes.length > 0 && (
                            <div className="mt-1 mb-1">
                              <span className="text-gray-500">Themes: </span>
                              <span className="text-gray-400">
                                {reflectionThemes.join(', ')}
                              </span>
                            </div>
                          )}
                          
                          {/* Display full reflection content if available */}
                          {(reflectionContentText || reflectionInsights.length > 0 || reflectionActions.length > 0 || reflectionEmotion) && (
                            <div className="mt-2 p-3 bg-gray-800/30 rounded border-l-2 border-[var(--color-accent)]/30 text-gray-300">
                              <div className="text-gray-500 text-xs mb-2">💭 Reflection</div>
                              {reflectionContentText && (
                                <div className="whitespace-pre-wrap text-sm mb-2">{reflectionContentText}</div>
                              )}
                              {reflectionInsights.length > 0 && (
                                <div className="mb-2">
                                  <div className="text-gray-500 text-[10px] uppercase tracking-wide mb-1">Insights</div>
                                  <ul className="list-disc list-inside text-xs space-y-1 text-gray-300">
                                    {reflectionInsights.map((insight, insightIdx) => (
                                      <li key={insightIdx}>{insight}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {reflectionActions.length > 0 && (
                                <div className="mb-2">
                                  <div className="text-gray-500 text-[10px] uppercase tracking-wide mb-1">Actions</div>
                                  <ul className="list-disc list-inside text-xs space-y-1 text-gray-300">
                                    {reflectionActions.map((action, actionIdx) => (
                                      <li key={actionIdx}>{action}</li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                              {reflectionEmotion && (
                                <div className="text-xs text-gray-400 space-y-1">
                                  {reflectionEmotion.primary && (
                                    <div>
                                      <span className="text-gray-500">Primary:</span> {reflectionEmotion.primary}
                                    </div>
                                  )}
                                  {intensityDisplay && (
                                    <div>
                                      <span className="text-gray-500">Intensity:</span> {intensityDisplay}
                                    </div>
                                  )}
                                  {Array.isArray(reflectionEmotion.secondary) && reflectionEmotion.secondary.length > 0 && (
                                    <div>
                                      <span className="text-gray-500">Secondary:</span> {reflectionEmotion.secondary.join(', ')}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )})}
          </div>
        ) : (
          <div className="bg-[var(--color-bg-elev-2)] p-4 rounded-md text-center border border-[var(--color-border)]">
            <p className="text-gray-400">No evolution history available yet</p>
            <p className="text-xs text-gray-500 mt-1">
              The persona will evolve autonomously as it learns from conversations and reflections
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PersonaEvolution;
