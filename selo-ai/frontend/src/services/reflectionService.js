/**
 * Reflection Service
 * 
 * Handles API communication with the reflection backend endpoints
 * and manages WebSocket connections for real-time reflection events.
 */

import { io } from 'socket.io-client';
import { getApiBaseUrl } from './config';

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
// Store callbacks for reconnection preservation
let storedOnReflectionUpdate = null;
let storedOnConnectionStatus = null;
const scheduleReinit = (reason = 'unknown') => {
  if (reinitScheduled || isConnecting) return;
  reinitScheduled = true;
  try { console.warn('[ReflectionSocket] scheduling re-init due to:', reason); } catch (_) {}
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
  }, 1000);
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
        console.error('Error in connection status callback:', error);
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
    try { console.warn('[ReflectionSocket] init aborted: missing userId'); } catch (_) {}
    return null;
  }
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
  
  // Define handlers before registration to avoid hoisting issues
  function handleGenerating(data) {
    // Debug logging removed for production
    const payload = {
      status: 'generating',
      type: data.reflection_type,
      data
    };
    onReflectionUpdate && onReflectionUpdate(payload);
    reflectionListeners.forEach(cb => {
      try { cb(payload); } catch (e) { console.error(e); }
    });
  }

  function handleGenerated(data) {
    const payload = {
      status: 'complete',
      type: data.reflection_type,
      reflectionId: data.reflection_id,
      data
    };
    onReflectionUpdate && onReflectionUpdate(payload);
    reflectionListeners.forEach(cb => {
      try { cb(payload); } catch (e) { console.error(e); }
    });
  }

  // Connect to reflection namespace after runtime config is available
  // Lazy-init: resolve base URL asynchronously and then connect
  getApiBaseUrl()
    .then((API_BASE_URL) => {
      // Log the resolved API base URL for diagnostics
      // Debug logging removed for production
      isConnecting = true;
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
        try { console.debug('[ReflectionSocket] connected', { clientInstanceId, sid: socket.id }); } catch (_) {}
        isConnecting = false;
        // Authenticate with the socket
        socket.emit('authenticate', { user_id: userId });
      });
      
      socket.on('disconnect', (reason) => {
        // Debug logging removed for production
        updateConnectionStatus('disconnected');
        onConnectionStatus && onConnectionStatus('disconnected'); // Backward compatibility
        try { console.debug('[ReflectionSocket] disconnected', { clientInstanceId, reason }); } catch (_) {}
        isConnecting = false;
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
        console.error('Reflection WebSocket connect_error:', err?.message || err, err);
        updateConnectionStatus('polling');
        isConnecting = false;
        // Common after restart or invalid sid; perform a clean re-init
        scheduleReinit('connect_error');
      });
      socket.io.on('reconnect_attempt', (attempt) => {
        console.warn('Reflection WebSocket reconnect_attempt', attempt);
        updateConnectionStatus('polling');
      });
      socket.io.on('reconnect_error', (err) => {
        console.error('Reflection WebSocket reconnect_error', err?.message || err, err);
        updateConnectionStatus('polling');
        isConnecting = false;
        scheduleReinit('reconnect_error');
      });
      socket.io.on('reconnect_failed', () => {
        console.error('Reflection WebSocket reconnect_failed');
        updateConnectionStatus('offline');
        isConnecting = false;
        scheduleReinit('reconnect_failed');
      });
      socket.on('ping', () => {
        console.debug('Reflection WebSocket ping');
      });
      socket.on('pong', (latency) => {
        console.debug('Reflection WebSocket pong', latency);
      });
      
      socket.on('authenticated', (data) => {
        // Debug logging removed for production
      });
      
      socket.on('error', (data) => {
        console.error('Reflection WebSocket error', data);
        // Some servers emit generic error on invalid sid; re-init defensively
        scheduleReinit('socket_error');
      });
      
      socket.on('reflection_generating', handleGenerating);
      socket.on('reflection_generated', handleGenerated);
    })
    .catch((e) => {
      console.error('Failed to initialize socket due to config error:', e);
      updateConnectionStatus('offline');
    });
  
  return socket;
};

/**
 * Disconnect WebSocket connection
 */
export const disconnectReflectionSocket = () => {
  if (socket) {
    intentionalDisconnect = true;
    socket.disconnect();
    socket = null;
    updateConnectionStatus('disconnected');
  }
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
      console.error('Socket authenticate failed', e);
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
    console.error('Error fetching reflections:', error);
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
    console.error(`Error fetching reflection ${reflectionId}:`, error);
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
    console.error('Error triggering reflection:', error);
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
    console.error('Error scheduling reflection:', error);
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
    console.error(`Error deleting reflection ${reflectionId}:`, error);
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
