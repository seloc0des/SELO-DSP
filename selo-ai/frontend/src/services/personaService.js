import axios from 'axios';
import { io } from 'socket.io-client';
import { getApiBaseUrl } from './config';
import { personaLogger as logger } from '../utils/logger';

// Socket.IO instance and cached base URL
let socket = null;
let cachedBaseUrl = null;
let baseUrlInflight = null;

const resolveBaseUrl = async () => {
  if (cachedBaseUrl) return cachedBaseUrl;
  if (baseUrlInflight) return baseUrlInflight;
  baseUrlInflight = getApiBaseUrl()
    .then((url) => {
      cachedBaseUrl = url;
      return url;
    })
    .finally(() => { baseUrlInflight = null; });
  return baseUrlInflight;
};

/**
 * Initialize Socket.IO connection for real-time persona updates
 */
const initSocket = async () => {
  if (!socket) {
    const API_URL = await resolveBaseUrl();
    socket = io(API_URL, {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      timeout: 20000,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 500,
      reconnectionDelayMax: 5000,
    });

    socket.on('connect', () => {
      // Debug logging removed for production
    });

    socket.on('disconnect', (reason) => {
      // Debug logging removed for production
    });
  }
  return socket;
};


/**
 * Get all personas for a user
 * @param {string} userId - User ID
 * @returns {Promise} - Promise with persona list
 */
const getPersonas = async (_userId) => {
  try {
    const API_URL = await resolveBaseUrl();
    // Single-user install: use default persona endpoint (ensures + returns persona)
    const response = await axios.get(`${API_URL}/api/persona/default`);
    const persona = response?.data?.data?.persona || null;
    const personas = persona ? [persona] : [];
    const count = persona ? 1 : 0;
    // Return both a simple top-level shape (for components) and the detailed payload for callers
    return {
      personas,
      count,
      success: true,
      message: persona ? 'Retrieved default persona' : 'No persona',
      data: { personas, count },
    };
  } catch (error) {
    logger.error('Failed to fetch personas:', error);
    return { personas: [] };
  }
};

/**
 * Ensure default persona for a user
 * @param {string} userId - User ID
 * @returns {Promise} - Promise with default persona
 */
const ensureDefaultPersona = async (_userId) => {
  try {
    const API_URL = await resolveBaseUrl();
    // Avoid system-key-protected endpoint; backend's GET /persona/default will ensure and return the default persona
    const response = await axios.get(`${API_URL}/api/persona/default`);
    return response.data;
  } catch (error) {
    logger.error('Failed to ensure default persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Get persona evolution history
 * @param {string} userId - User ID
 * @param {string} personaId - Persona ID
 * @returns {Promise} - Promise with persona evolution history
 */
const getPersonaEvolutionHistory = async (_userId, personaId) => {
  try {
    const API_URL = await resolveBaseUrl();
    // Backend route: GET /persona/history/{persona_id}
    const response = await axios.get(`${API_URL}/api/persona/history/${encodeURIComponent(personaId)}`);
    // Backend returns: { success, message, persona_id, data: { persona, evolutions, evolution_count, trait_histories } }
    const data = (response && response.data && response.data.data) ? response.data.data : {};
    return { history: data.evolutions || [] };
  } catch (error) {
    logger.error('Failed to fetch persona evolution history:', error);
    return { history: [] };
  }
};

/**
 * Get persona system prompt
 * @param {string} userId - User ID
 * @param {string} personaId - Persona ID
 * @returns {Promise} - Promise with persona system prompt
 */
const getPersonaSystemPrompt = async (_userId, personaId) => {
  try {
    const API_URL = await resolveBaseUrl();
    // Backend route: GET /persona/system-prompt/{persona_id}
    const response = await axios.get(`${API_URL}/api/persona/system-prompt/${encodeURIComponent(personaId)}`);
    // Backend returns: { success, message, persona_id, data: { system_prompt } }
    const data = (response && response.data && response.data.data) ? response.data.data : {};
    return { prompt: data.system_prompt || '' };
  } catch (error) {
    logger.error('Failed to fetch persona system prompt:', error);
    return { prompt: '' };
  }
};

/**
 * Get default persona (single-user convenience)
 * @returns {Promise<{success: boolean, data?: {id: string, user_id: string, persona: object}}>}
 */
const getDefaultPersona = async () => {
  try {
    const API_URL = await resolveBaseUrl();
    const response = await axios.get(`${API_URL}/api/persona/default`);
    return response.data;
  } catch (error) {
    logger.error('Failed to fetch default persona:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Get persona presentation content (first introduction and latest session summary)
 * @param {string} userId - User ID (unused on backend for single-user, retained for API parity)
 * @param {string} personaId - Persona ID
 * @returns {{success:boolean, data:{first_intro:string, last_session_summary:string}}}
 */
const getPersonaPresentation = async (_userId, personaId) => {
  try {
    const API_URL = await resolveBaseUrl();
    const response = await axios.get(`${API_URL}/api/persona/presentation/${encodeURIComponent(personaId)}`, {
      timeout: 15000
    });
    return response.data;
  } catch (error) {
    logger.error('Failed to fetch persona presentation:', error);
    return { success: false, data: { first_intro: '', last_session_summary: '', first_thoughts: '' } };
  }
};

/**
 * Get persona traits
 * @param {string} _userId - User ID (unused, retained for backward compatibility)
 * @param {string} _personaId - Persona ID (unused, we always use default persona)
 * @returns {Promise<{traits: Array, count?: number, persona?: Object}>} - Promise with persona traits and metadata
 */
const getPersonaTraits = async (_userId, _personaId) => {
  try {
    const API_URL = await resolveBaseUrl();
    // Always use the default persona endpoint for single-user systems
    const response = await axios.get(`${API_URL}/api/persona/default/traits`);
    
    if (response.data && response.data.success) {
      return {
        traits: response.data.data?.traits || [],
        count: response.data.data?.count || 0,
        persona: response.data.data?.persona || null
      };
    }
    
    logger.warn('Unexpected response format:', response.data);
    return { traits: [], count: 0 };
  } catch (error) {
    logger.error('Failed to fetch persona traits:', error);
    // Still return empty but include more debugging info
    return { 
      traits: [], 
      count: 0,
      error: error.message 
    };
  }
};

/**
 * Get default persona traits (single-user convenience)
 * @param {string|null} category - Optional category filter
 */
const getDefaultPersonaTraits = async (category = null) => {
  try {
    const API_URL = await resolveBaseUrl();
    const response = await axios.get(`${API_URL}/api/persona/default/traits`, {
      params: category ? { category } : {}
    });
    // Backend returns: { success, message, persona_id, data: { traits, count, persona } }
    const data = (response && response.data && response.data.data) ? response.data.data : {};
    return { traits: data.traits || [], count: data.count || 0, persona: data.persona || null };
  } catch (error) {
    logger.error('Failed to fetch default persona traits:', error);
    return { traits: [] };
  }
};

/**
 * Get per-trait history for a persona
 * @param {string} personaId
 * @param {string} traitName
 * @param {object} opts - optional { category, limit }
 */
const getTraitHistory = async (personaId, traitName, opts = {}) => {
  try {
    const API_URL = await resolveBaseUrl();
    const params = {};
    if (opts.category) params.trait_category = opts.category;
    if (opts.limit) params.limit = opts.limit;
    const response = await axios.get(`${API_URL}/api/persona/traits/${personaId}/${encodeURIComponent(traitName)}/history`, {
      params
    });
    return response.data;
  } catch (error) {
    logger.error('Failed to fetch trait history:', error);
    return { history: [] };
  }
};

// Export all service functions
const personaService = {
  initSocket,
  getPersonas,
  ensureDefaultPersona,
  getDefaultPersona,
  getPersonaEvolutionHistory,
  getPersonaSystemPrompt,
  getPersonaPresentation,
  getPersonaTraits,
  getDefaultPersonaTraits,
  getTraitHistory,
};

export default personaService;
