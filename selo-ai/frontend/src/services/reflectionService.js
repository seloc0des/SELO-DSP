/**
 * Reflection Service
 * 
 * Handles API communication with the reflection backend endpoints
 * and manages WebSocket connections for real-time reflection events.
 */

import { io } from 'socket.io-client';
import { getApiBaseUrl } from './config';
import { reflectionLogger as logger } from '../utils/logger';

// Socket.IO connection and status management
let socket = null;
let connectionStatus = 'disconnected';
let connectionListeners = new Set();
let reflectionListeners = new Set();
let currentUserId = null;
let isConnecting = false; // prevent racing inits
const clientInstanceId = `reflect-${Math.random().toString(36).slice(2)}-${Date.now()}`;
// Track whether a disconnect was initiated intentionally by the client (e.g., unmount)
let intentionalDisconnect = false;
// Re-init throttling to avoid reconnection storms
let reinitScheduled = false;
let inflightInit = null;
let reinitDelayMs = 1000;
// Store callbacks for reconnection preservation
let storedOnReflectionUpdate = null;
let storedOnConnectionStatus = null;
const scheduleReinit = (reason = 'unknown') => {
  if (reinitScheduled || isConnecting) return;
  if (!currentUserId) return;
  reinitScheduled = true;
  logger.warn('Scheduling re-init due to:', reason);
  const delay = reinitDelayMs;
  reinitDelayMs = Math.min(reinitDelayMs * 2, 10000);
  setTimeout(() => {
    try {
      if (socket) {
        try { socket.removeAllListeners(); } catch (_) {}
        try { socket.disconnect(); } catch (_) {}
      }
      socket = null;
      // Re-initialize with the last known user id and preserved callbacks
      // Listeners are preserved via reflectionListeners set AND stored callbacks
      if (currentUserId) {
        initReflectionSocket(currentUserId, storedOnReflectionUpdate, storedOnConnectionStatus);
      }
    } finally {
      reinitScheduled = false;
    }
  }, delay);
};

/**
 * Subscribe to connection status changes
 * @param {function} callback - Callback function to receive status updates
 * @returns {function} Unsubscribe function
 */
export const subscribeToConnectionStatus = (callback) => {
  connectionListeners.add(callback);
  // Immediately call with current status
  callback(connectionStatus);
  
  // Return unsubscribe function
  return () => {
    connectionListeners.delete(callback);
  };
};

/**
 * Get current connection status
 * @returns {string} Current connection status
 */
export const getConnectionStatus = () => connectionStatus;

/**
 * Update connection status and notify all listeners
 * @param {string} status - New connection status
 */
const updateConnectionStatus = (status) => {
  if (connectionStatus !== status) {
    connectionStatus = status;
    connectionListeners.forEach(callback => {
      try {
        callback(status);
      } catch (error) {
        logger.error('Error in connection status callback:', error);
      }
    });
  }
};

/**
 * Initialize WebSocket connection for real-time reflection events
 * @param {string} userId - User ID for authentication
 * @param {function} onReflectionUpdate - Callback when reflection updates occur
 * @param {function} onConnectionStatus - Callback for connection status changes (deprecated - use subscribeToConnectionStatus)
 * @returns {object} Socket instance
 */
export const initReflectionSocket = (userId, onReflectionUpdate, onConnectionStatus) => {
  // Do not start a socket connection without a valid user id
  if (!userId) {
    logger.warn('Init aborted: missing userId');
    return null;
  }
  intentionalDisconnect = false;
  updateConnectionStatus('connecting');
  currentUserId = userId;
  // Store callbacks for reconnection preservation
  if (onReflectionUpdate) storedOnReflectionUpdate = onReflectionUpdate;
  if (onConnectionStatus) storedOnConnectionStatus = onConnectionStatus;
  
  // If a socket is already connecting or connected, avoid racing a new init
  if (socket && (socket.connected || socket.connecting || isConnecting)) {
    currentUserId = userId;
    try { socket.emit('authenticate', { user_id: userId }); } catch (_) {}
    return socket;
  }
  // Close existing idle/closed connection if any
  if (socket) {
    try { socket.removeAllListeners(); } catch (_) {}
    try { socket.disconnect(); } catch (_) {}
  }

  if (inflightInit) {
    currentUserId = userId;
    return socket;
  }
  isConnecting = true;
  
  // Define handlers before registration to avoid hoisting issues
  function handleGenerating(data) {
    // Debug logging removed for production
    const payload = {
      status: 'generating',
      type: data?.reflection_type || data?.type,
      data
    };
    onReflectionUpdate && onReflectionUpdate(payload);
    reflectionListeners.forEach(cb => {
      try { cb(payload); } catch (e) { logger.error('Reflection listener error:', e); }
    });
  }

  function handleGenerated(data) {
    const payload = {
      status: 'complete',
      type: data?.reflection_type || data?.type,
      reflectionId: data?.reflection_id || data?.reflectionId || data?.id,
      data
    };
    onReflectionUpdate && onReflectionUpdate(payload);
    reflectionListeners.forEach(cb => {
      try { cb(payload); } catch (e) { logger.error('Reflection listener error:', e); }
    });
  }

  // Connect to reflection namespace after runtime config is available
  // Lazy-init: resolve base URL asynchronously and then connect
  inflightInit = getApiBaseUrl()
    .then((API_BASE_URL) => {
      if (!currentUserId || currentUserId !== userId || intentionalDisconnect) {
        isConnecting = false;
        inflightInit = null;
        intentionalDisconnect = false;
        updateConnectionStatus('disconnected');
        return null;
      }
      // Log the resolved API base URL for diagnostics
      // Debug logging removed for production
      socket = io(`${API_BASE_URL}/reflection`, {
        // Explicitly match backend Socket.IO path and force pure WebSocket (no polling)
        path: '/socket.io',
        transports: ['websocket'],
        upgrade: false,
        autoConnect: true,
        reconnection: true,
        // Conservative timeouts/retry tuning
        timeout: 60000, // 60s connection timeout for slow networks/servers under load
        reconnectionAttempts: Infinity,
        reconnectionDelay: 500,
        reconnectionDelayMax: 5000,
      });
      currentUserId = userId;
      
      // Set up event handlers after socket is created
      socket.on('connect', () => {
        // Debug logging removed for production
        updateConnectionStatus('connected');
        onConnectionStatus && onConnectionStatus('connected'); // Backward compatibility
        logger.debug('Connected', { clientInstanceId, sid: socket.id });
        isConnecting = false;
        inflightInit = null;
        reinitDelayMs = 1000;
        // Authenticate with the socket
        socket.emit('authenticate', { user_id: userId });
      });
      
      socket.on('disconnect', (reason) => {
        // Debug logging removed for production
        updateConnectionStatus('disconnected');
        onConnectionStatus && onConnectionStatus('disconnected'); // Backward compatibility
        logger.debug('Disconnected', { clientInstanceId, reason });
        isConnecting = false;
        inflightInit = null;
        // If this was an intentional client disconnect (component unmount/navigation), do not re-init
        if (intentionalDisconnect) {
          intentionalDisconnect = false; // reset flag
          return;
        }
        // If server restarted or transport closed, schedule a clean re-init
        if (reason === 'transport error' || reason === 'io server disconnect' || reason === 'ping timeout') {
          scheduleReinit(`disconnect:${reason}`);
        }
      });

      // Additional diagnostics
      socket.on('connect_error', (err) => {
        logger.error('WebSocket connect_error:', err?.message || err);
        updateConnectionStatus('polling');
        isConnecting = false;
        inflightInit = null;
        // Common after restart or invalid sid; perform a clean re-init
        scheduleReinit('connect_error');
      });
      socket.io.on('reconnect_attempt', (attempt) => {
        logger.warn('WebSocket reconnect_attempt', attempt);
        updateConnectionStatus('polling');
      });
      socket.io.on('reconnect_error', (err) => {
        logger.error('WebSocket reconnect_error', err?.message || err);
        updateConnectionStatus('polling');
        isConnecting = false;
        inflightInit = null;
        scheduleReinit('reconnect_error');
      });
      socket.io.on('reconnect_failed', () => {
        logger.error('WebSocket reconnect_failed');
        updateConnectionStatus('offline');
        isConnecting = false;
        inflightInit = null;
        scheduleReinit('reconnect_failed');
      });
      socket.on('ping', () => {
        logger.debug('WebSocket ping');
      });
      socket.on('pong', (latency) => {
        logger.debug('WebSocket pong', latency);
      });
      
      socket.on('authenticated', (data) => {
        // Debug logging removed for production
      });
      
      socket.on('error', (data) => {
        logger.error('WebSocket error', data);
        inflightInit = null;
        // Some servers emit generic error on invalid sid; re-init defensively
        scheduleReinit('socket_error');
      });
      
      socket.on('reflection_generating', (data) => {
        logger.info('reflection_generating event', data);
        handleGenerating(data);
      });
      socket.on('reflection_generated', (data) => {
        logger.info('reflection_generated event', data);
        handleGenerated(data);
      });
    })
    .catch((e) => {
      logger.error('Failed to initialize socket due to config error:', e);
      updateConnectionStatus('offline');
      isConnecting = false;
      inflightInit = null;
      return null;
    });
  
  return socket;
};

/**
 * Disconnect WebSocket connection
 */
export const disconnectReflectionSocket = () => {
  intentionalDisconnect = true;
  currentUserId = null;
  if (socket) {
    socket.disconnect();
  }
  socket = null;
  inflightInit = null;
  isConnecting = false;
  reinitDelayMs = 1000;
  updateConnectionStatus('disconnected');
};

// Ensure we have a connected socket for a given user
const ensureSocket = (userId) => {
  if (!socket) {
    // Do not initialize without a valid user id; wait until it becomes available
    if (userId) {
      initReflectionSocket(userId, null, null);
    } else {
      return;
    }
  } else if (userId && currentUserId !== userId) {
    try {
      socket.emit('authenticate', { user_id: userId });
      currentUserId = userId;
    } catch (e) {
      logger.error('Socket authenticate failed', e);
    }
  }
};

/**
 * Fetch list of reflections for a user
 * @param {string} userId - User ID
 * @param {string} reflectionType - Optional reflection type filter
 * @param {number} limit - Max number of reflections to fetch
 * @param {number} offset - Pagination offset
 * @returns {Promise<object>} Reflection list response
 */
export const fetchReflections = async (userId, reflectionType = null, limit = 10, offset = 0) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    let url = `${API_BASE_URL}/api/reflections/list?user_profile_id=${userId}&limit=${limit}&offset=${offset}`;
    
    if (reflectionType) {
      url += `&reflection_type=${reflectionType}`;
    }
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    logger.error('Error fetching reflections:', error);
    throw error;
  }
};

/**
 * Fetch a specific reflection by ID
 * @param {string} reflectionId - Reflection ID
 * @returns {Promise<object>} Reflection data
 */
export const fetchReflection = async (reflectionId) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    const response = await fetch(`${API_BASE_URL}/api/reflections/${reflectionId}`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    logger.error(`Error fetching reflection ${reflectionId}:`, error);
    throw error;
  }
};

/**
 * [INTERNAL USE ONLY] Trigger a new reflection to be generated
 * This function is only for internal system use. User-triggered reflections are not supported.
 * Reflections are generated autonomously by the system scheduler.
 * 
 * @private
 * @param {string} userId - User ID
 * @param {string} reflectionType - Type of reflection to generate
 * @param {string[]} memoryIds - Optional specific memory IDs to include
 * @returns {Promise<object>} Generation status
 */
const _triggerReflection = async (userId, reflectionType, memoryIds = null) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    const response = await fetch(`${API_BASE_URL}/api/reflections/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_profile_id: userId,
        reflection_type: reflectionType,
        memory_ids: memoryIds,
        trigger_source: 'system'
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    logger.error('Error triggering reflection:', error);
    throw error;
  }
};

/**
 * [INTERNAL USE ONLY] Schedule a reflection for later generation
 * This function is only for internal system use. User scheduling is not supported.
 * Reflections are scheduled autonomously by the system.
 * 
 * @private
 * @param {string} userId - User ID
 * @param {string} reflectionType - Type of reflection to schedule
 * @returns {Promise<object>} Schedule status
 */
const _scheduleReflection = async (userId, reflectionType) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    const response = await fetch(`${API_BASE_URL}/api/reflections/schedule/${reflectionType}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_profile_id: userId
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    logger.error('Error scheduling reflection:', error);
    throw error;
  }
};

/**
 * [INTERNAL USE ONLY] Delete a reflection by ID
 * This function is only for internal system use.
 * To maintain a complete audit trail, users cannot delete reflections.
 * 
 * @private
 * @param {string} reflectionId - Reflection ID
 * @returns {Promise<object>} Deletion status
 */
const _deleteReflection = async (reflectionId) => {
  try {
    const API_BASE_URL = await getApiBaseUrl();
    const response = await fetch(`${API_BASE_URL}/api/reflections/${reflectionId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    logger.error(`Error deleting reflection ${reflectionId}:`, error);
    throw error;
  }
};

// Named service object used by components
export const reflectionService = {
  getReflections: async (userId, reflectionType = null, limit = 20, offset = 0) => {
    const resp = await fetchReflections(userId, reflectionType, limit, offset);
    // Normalize to an array for consumers like ReflectionPanel
    if (Array.isArray(resp)) return resp;
    return resp?.reflections || [];
  },

  onReflectionGenerated: (callback, userId) => {
    ensureSocket(userId);
    reflectionListeners.add(callback);
    return () => reflectionListeners.delete(callback);
  },

  subscribeToConnectionStatus,
  getConnectionStatus,
  disconnect: () => disconnectReflectionSocket(),
  init: (userId, onUpdate, onStatus) => initReflectionSocket(userId, onUpdate, onStatus),
};
