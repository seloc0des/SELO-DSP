import React, { useState, useEffect, useMemo } from 'react';
import personaService from '../../services/personaService';
import { personaLogger as logger } from '../../utils/logger';
import PersonaTraitRadar from './PersonaTraitRadar';

/**
 * Component for displaying a single persona card
 * Shows basic persona information and allows selection
 */
const PersonaCard = ({ persona, isSelected, onClick }) => {
  const [traits, setTraits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [presentation, setPresentation] = useState({ first_intro: '', first_thoughts: '', last_session_summary: '' });
  const [loadingPresentation, setLoadingPresentation] = useState(false);
  
  useEffect(() => {
    const fetchData = async () => {
      if (!persona?.id) return;
      
      setLoading(true);
      try {
        // First get the current default persona to ensure we have the latest ID
        const defaultPersona = await personaService.getDefaultPersona();
        if (!defaultPersona?.success || !defaultPersona.data?.persona?.id) {
          throw new Error('Could not fetch default persona');
        }
        
        // Get traits for the current default persona
        const traitsResponse = await personaService.getPersonaTraits(
          defaultPersona.data.persona.user_id, 
          defaultPersona.data.persona.id
        );
        const traitsArr = (traitsResponse && Array.isArray(traitsResponse.traits)) 
          ? traitsResponse.traits 
          : [];
        setTraits(traitsArr);
        
        // Fetch persona presentation
        setLoadingPresentation(true);
        const presentationResp = await personaService.getPersonaPresentation(
          defaultPersona.data.persona.user_id, 
          defaultPersona.data.persona.id
        );
        
        const payload = (presentationResp?.data) 
          ? presentationResp.data 
          : {};
          
        setPresentation({
          first_intro: typeof payload.first_intro === 'string' ? payload.first_intro : '',
          first_thoughts: typeof payload.first_thoughts === 'string' ? payload.first_thoughts : '',
          last_session_summary: typeof payload.last_session_summary === 'string' 
            ? payload.last_session_summary 
            : '',
        });
      } catch (error) {
        logger.error('Error in PersonaCard data fetch:', error);
        setTraits([]);
        setPresentation({ first_intro: '', first_thoughts: '', last_session_summary: '' });
      } finally {
        setLoading(false);
        setLoadingPresentation(false);
      }
    };
    
    if (persona?.id) {
      fetchData();
    }
  }, [persona]);
  
  const topSixRadarTraits = useMemo(() => {
    const sortable = (traits || [])
      .filter(t => typeof t.value === 'number')
      .sort((a, b) => {
        const bc = typeof b.confidence === 'number' ? b.confidence : 0;
        const ac = typeof a.confidence === 'number' ? a.confidence : 0;
        if (bc !== ac) return bc - ac;
        const bv = typeof b.value === 'number' ? b.value : 0;
        const av = typeof a.value === 'number' ? a.value : 0;
        return bv - av;
      });
    return sortable.slice(0, 6).map(t => ({
      name: t.name,
      value: t.value,
      category: t.category,
    }));
  }, [traits]);
  
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
  
  // Traits are visualized via PersonaTraitRadar below
    
  // Color styling based on selection state (modern dark theme)
  const cardClasses = isSelected 
    ? 'border-[var(--color-accent)] bg-[var(--color-bg-elev-2)]'
    : 'border-[var(--color-border)] bg-[var(--color-bg-elev-1)] hover:bg-[var(--color-bg-elev-2)]';
  
  return (
    <div 
      className={`rounded-md border p-4 cursor-pointer transition-all ${cardClasses}`}
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-2">
        <h4 className="text-lg font-medium text-[var(--color-text-primary)]">
          {persona?.name || persona?.data?.name || 'SELO'}
        </h4>
        <span className="text-xs px-2 py-1 rounded bg-[var(--color-bg-elev-2)] text-[var(--color-text-secondary)] border border-[var(--color-border)]">
          v{persona.version || '1.0'}
        </span>
      </div>
      
      <div className="mb-3">
        <div className="mb-2">
          <span className="text-xs text-[var(--color-text-muted)] uppercase tracking-wide">First Thoughts</span>
        </div>
        {loadingPresentation ? (
          <div className="h-10 animate-pulse bg-[var(--color-bg-elev-2)] rounded" />
        ) : (
          <p className="text-sm text-[var(--color-text-secondary)] italic whitespace-pre-wrap">
            {presentation.first_thoughts 
              || presentation.first_intro 
              || persona.boot_directive 
              || persona.description 
              || 'No first thoughts available'}
          </p>
        )}
      </div>
      
      {loading ? (
        <div className="h-10 animate-pulse bg-[var(--color-bg-elev-2)] rounded"></div>
      ) : (
        <div className="mb-3">
          {/* Read-only radar visualization of traits (no controls) */}
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-[var(--color-text-secondary)]">Top 6 Weighted Traits</h4>
          </div>
          <PersonaTraitRadar traits={topSixRadarTraits} size={280} />
        </div>
      )}
      
      <div className="mt-3 text-xs text-[var(--color-text-muted)]">
        <div>Created: {formatDate(persona.creation_date)}</div>
        <div>Last evolved: {formatDate(persona.last_modified)}</div>
      </div>
    </div>
  );
};

export default PersonaCard;
