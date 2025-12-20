/**
 * Agent State Service
 * 
 * Handles API communication for emergent agent state data:
 * - Affective state (energy, stress, confidence, mood)
 * - Goals (active goals from reflections)
 * - Meta-directives (self-assigned tasks)
 * - Autobiographical episodes (significant moments)
 */

import { getApiBaseUrl } from './config';

/**
 * Get current affective state for a persona
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @returns {Promise<object>} Affective state data
 */
export const getAffectiveState = async (personaId = null) => {
  try {
    const apiBaseUrl = await getApiBaseUrl();
    const url = personaId 
      ? `${apiBaseUrl}/api/agent-state/affective?persona_id=${personaId}`
      : `${apiBaseUrl}/api/agent-state/affective`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch affective state: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching affective state:', error);
    throw error;
  }
};

/**
 * Get active goals for a persona
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @returns {Promise<object>} Goals data
 */
export const getGoals = async (personaId = null) => {
  try {
    const apiBaseUrl = await getApiBaseUrl();
    const url = personaId 
      ? `${apiBaseUrl}/api/agent-state/goals?persona_id=${personaId}`
      : `${apiBaseUrl}/api/agent-state/goals`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch goals: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching goals:', error);
    throw error;
  }
};

/**
 * Get plan steps for a persona
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @returns {Promise<object>} Plan steps data
 */
export const getPlanSteps = async (personaId = null) => {
  try {
    const apiBaseUrl = await getApiBaseUrl();
    const url = personaId 
      ? `${apiBaseUrl}/api/agent-state/plan-steps?persona_id=${personaId}`
      : `${apiBaseUrl}/api/agent-state/plan-steps`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch plan steps: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching plan steps:', error);
    throw error;
  }
};

/**
 * Get meta-directives for a persona
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @returns {Promise<object>} Meta-directives data
 */
export const getMetaDirectives = async (personaId = null) => {
  try {
    const apiBaseUrl = await getApiBaseUrl();
    const url = personaId 
      ? `${apiBaseUrl}/api/agent-state/meta-directives?persona_id=${personaId}`
      : `${apiBaseUrl}/api/agent-state/meta-directives`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch meta-directives: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching meta-directives:', error);
    throw error;
  }
};

/**
 * Get autobiographical episodes for a persona
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @param {number} limit - Maximum number of episodes to fetch
 * @returns {Promise<object>} Episodes data
 */
export const getEpisodes = async (personaId = null, limit = 5) => {
  try {
    const apiBaseUrl = await getApiBaseUrl();
    const url = personaId 
      ? `${apiBaseUrl}/api/agent-state/episodes?persona_id=${personaId}&limit=${limit}`
      : `${apiBaseUrl}/api/agent-state/episodes?limit=${limit}`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch episodes: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching episodes:', error);
    throw error;
  }
};

/**
 * Fetch all agent state data at once
 * @param {string} personaId - Persona ID (optional, uses default if not provided)
 * @returns {Promise<object>} Combined agent state data
 */
export const getAllAgentState = async (personaId = null) => {
  try {
    const [affective, goals, directives, episodes] = await Promise.all([
      getAffectiveState(personaId),
      getGoals(personaId),
      getMetaDirectives(personaId),
      getEpisodes(personaId, 5)
    ]);
    
    return {
      affective,
      goals,
      directives,
      episodes
    };
  } catch (error) {
    console.error('Error fetching all agent state:', error);
    throw error;
  }
};
