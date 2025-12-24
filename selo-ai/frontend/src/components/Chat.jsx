import React, { useState, useRef, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import ReflectionPanel from './ReflectionPanel';
import { reflectionService } from '../services/reflectionService';
import { getApiBaseUrl } from '../services/config';
import { formatRelativeTime } from '../utils/dateFormatter';
import { io } from 'socket.io-client';
import { chatLogger as logger } from '../utils/logger';
import ErrorBoundary from './ErrorBoundary';

// Runtime-resolved API base URL
const trimTrailingSlash = (s) => s.replace(/\/$/, '');
// Enforce strict reflection-first UX (no assistant release without matching reflection)
const STRICT_REFLECTION_FIRST = true;

const Chat = ({ userId }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiBase, setApiBase] = useState(null);
  // Map of turnId -> assistant message content waiting for matching reflection
  const pendingAnswersRef = useRef(new Map());
  const reflectionsSeenRef = useRef(new Set());
  const [sessionId, setSessionId] = useState(null);
  // Track per-turn timeouts for releasing pending assistant messages if reflection is delayed
  const pendingTimeoutsRef = useRef(new Map());
  const chatSocketRef = useRef(null);
  const streamingBuffersRef = useRef(new Map());
  const releasedTurnsRef = useRef(new Set());

  const setPendingAnswer = useCallback((turnId, { content, timestamp } = {}) => {
    if (!turnId) return;
    const existing = pendingAnswersRef.current.get(turnId) || {};
    const resolvedContent = content !== undefined ? content : existing.content;
    const resolvedTimestamp = timestamp ?? existing.timestamp ?? new Date().toISOString();
    pendingAnswersRef.current.set(turnId, {
      content: resolvedContent !== undefined ? resolvedContent : '',
      timestamp: resolvedTimestamp,
    });
    releasedTurnsRef.current.delete(turnId);
  }, []);

  const flushPendingAnswer = useCallback((turnId, overrides = {}) => {
    if (!turnId || releasedTurnsRef.current.has(turnId)) return false;
    const pending = pendingAnswersRef.current.get(turnId);
    if (!pending && overrides.content === undefined && overrides.timestamp === undefined) {
      return false;
    }
    const content = overrides.content !== undefined ? overrides.content : (pending?.content ?? '');
    const timestamp = overrides.timestamp ?? pending?.timestamp ?? new Date().toISOString();
    pendingAnswersRef.current.delete(turnId);
    const timeoutId = pendingTimeoutsRef.current.get(turnId);
    if (timeoutId) {
      clearTimeout(timeoutId);
      pendingTimeoutsRef.current.delete(turnId);
    }
    streamingBuffersRef.current.delete(turnId);
    releasedTurnsRef.current.add(turnId);
    setMessages(prev => [...prev, {
      role: 'assistant',
      content,
      timestamp,
    }]);
    return true;
  }, [setMessages]);
  
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup pending timeouts on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      // Clear all pending timeouts when component unmounts
      pendingTimeoutsRef.current.forEach(timeoutId => {
        clearTimeout(timeoutId);
      });
      pendingTimeoutsRef.current.clear();
      streamingBuffersRef.current.clear();
      pendingAnswersRef.current.clear();
      reflectionsSeenRef.current.clear();
      releasedTurnsRef.current.clear();
      if (chatSocketRef.current) {
        try { chatSocketRef.current.disconnect(); } catch (_) {}
        chatSocketRef.current = null;
      }
    };
  }, []);

  // Resolve API base URL at runtime via backend /config.json (with fallbacks inside getApiBaseUrl)
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const base = await getApiBaseUrl();
        if (mounted) setApiBase(base);
      } catch (e) {
        logger.warn('Failed to resolve API base URL:', e);
        if (mounted) setApiBase('http://localhost:8000');
      }
    })();
    return () => { mounted = false; };
  }, []);

  // Establish a single, consistent session id across the app. Prefer App's userId prop.
  useEffect(() => {
    // If App provided a userId, use it and persist to localStorage for continuity
    if (userId && typeof userId === 'string') {
      setSessionId(userId);
      try { 
        localStorage.setItem('selo_ai_session_id', userId); 
      } catch (storageErr) {
        logger.warn('localStorage unavailable for session persistence (private browsing?)', storageErr);
      }
      return;
    }
    // Fallback to existing persisted session id or create a new one
    let sid = null;
    try { 
      sid = localStorage.getItem('selo_ai_session_id'); 
    } catch (storageErr) { 
      logger.warn('localStorage read failed (private browsing mode?)', storageErr);
      sid = null; 
    }
    if (!sid) {
      sid = uuidv4();
      try { 
        localStorage.setItem('selo_ai_session_id', sid); 
      } catch (storageErr) {
        logger.warn('localStorage write failed - session will not persist across refreshes', storageErr);
      }
    }
    setSessionId(sid);
  }, [userId]);

  // Load conversation history when sessionId is available
  useEffect(() => {
    if (!sessionId || !apiBase) return;
    
    const loadConversationHistory = async () => {
      try {
        const response = await fetch(`${trimTrailingSlash(apiBase)}/conversations/history?session_id=${sessionId}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.messages && Array.isArray(data.messages)) {
            // Convert backend message format to frontend format
            const formattedMessages = data.messages.map(msg => ({
              role: msg.role === 'user' ? 'user' : 'assistant',
              content: msg.content,
              timestamp: msg.timestamp || null,
            }));
            setMessages(formattedMessages);
          }
        }
      } catch (error) {
        logger.warn('Failed to load conversation history:', error);
        // Don't show error to user, just start with empty conversation
      }
    };
    
    loadConversationHistory();
  }, [sessionId, apiBase]);

  // Subscribe to reflections: show a visible "Reflecting…" indicator during generation and
  // release pending assistant messages when matching turn_id arrives
  useEffect(() => {
    if (!sessionId) return;
    const unsubscribe = reflectionService.onReflectionGenerated((evt) => {
      // Reflection service emits both generating and generated events to this listener.
      // evt.status can be 'generating' | 'complete'.
      const base = evt || {};
      const payload = base?.data || base;
      if (!payload) return;
      const { turn_id, user_profile_id } = payload;
      if (!turn_id) return;
      if (user_profile_id && user_profile_id !== sessionId) return;

      if (base.status === 'complete') {
        reflectionsSeenRef.current.add(turn_id);
        const released = flushPendingAnswer(turn_id);
        if (!released && pendingAnswersRef.current.size === 1) {
          // Fallback: if exactly one pending answer exists for this session but the turn_id didn't match,
          // release it to avoid a stuck UI due to rare id mismatch races.
          logger.warn('Releasing unmatched pending assistant due to single-pending fallback');
          const [[k, v]] = Array.from(pendingAnswersRef.current.entries());
          flushPendingAnswer(k, {
            content: v?.content,
            timestamp: v?.timestamp,
          });
        }
      }
    }, sessionId);

    return () => {
      if (typeof unsubscribe === 'function') unsubscribe();
    };
  }, [sessionId, flushPendingAnswer]);

  useEffect(() => {
    if (!sessionId || !apiBase) return;

    const socket = io(`${trimTrailingSlash(apiBase)}/chat`, {
      path: '/socket.io',
      transports: ['websocket'],
      upgrade: false,
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 500,
      reconnectionDelayMax: 5000,
    });
    chatSocketRef.current = socket;

    socket.on('connect', () => {
      try {
        socket.emit('join', { user_id: sessionId });
      } catch (_) {}
    });

    socket.on('chat_chunk', (payload = {}) => {
      const turnId = payload.turn_id || payload.turnId;
      const chunk = payload.chunk;
      if (!turnId || typeof chunk !== 'string' || chunk.length === 0) return;

      const previous = streamingBuffersRef.current.get(turnId) || '';
      const combined = previous + chunk;
      streamingBuffersRef.current.set(turnId, combined);

      setPendingAnswer(turnId, {
        content: combined,
        timestamp: payload.timestamp || new Date().toISOString(),
      });

      if (payload.final && reflectionsSeenRef.current.has(turnId)) {
        flushPendingAnswer(turnId, {
          content: combined,
          timestamp: payload.timestamp,
        });
      }
    });

    socket.on('chat_complete', (payload = {}) => {
      const turnId = payload.turn_id || payload.turnId;
      if (!turnId) return;

      const content = typeof payload.content === 'string' ? payload.content : '';
      const timestamp = payload.timestamp || new Date().toISOString();

      streamingBuffersRef.current.delete(turnId);
      setPendingAnswer(turnId, { content, timestamp });

      if (reflectionsSeenRef.current.has(turnId)) {
        flushPendingAnswer(turnId, { content, timestamp });
      }
    });

    return () => {
      socket.off('connect');
      socket.off('chat_chunk');
      socket.off('chat_complete');
      try { socket.disconnect(); } catch (_) {}
      if (chatSocketRef.current === socket) {
        chatSocketRef.current = null;
      }
      streamingBuffersRef.current.clear();
    };
  }, [sessionId, apiBase, flushPendingAnswer, setPendingAnswer]);

  const clearConversation = () => {
    setMessages([]);
    // Optionally generate a new session ID to start fresh
    const newSessionId = uuidv4();
    setSessionId(newSessionId);
    try { 
      localStorage.setItem('selo_ai_session_id', newSessionId); 
    } catch (_) {}
    pendingAnswersRef.current.clear();
    reflectionsSeenRef.current.clear();
    releasedTurnsRef.current.clear();
    streamingBuffersRef.current.clear();
    pendingTimeoutsRef.current.forEach(timeoutId => {
      clearTimeout(timeoutId);
    });
    pendingTimeoutsRef.current.clear();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading || !apiBase || !sessionId) return;

    const nowIso = new Date().toISOString();
    const userMessage = { role: 'user', content: input, timestamp: nowIso };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const CHAT_ENDPOINT = `${trimTrailingSlash(apiBase)}/chat`;
      // Debug logging removed for production
      const response = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          prompt: input,
        }),
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        logger.error('/chat non-OK', response.status, text);
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      // Debug logging removed for production
      const turnId = data.turn_id;
      const assistantContent = data.response;
      const assistantTimestamp = data.timestamp || new Date().toISOString();
      // Prefer showing reflection first. Gate assistant until matching reflection arrives,
      // but fall back after a short timeout to prevent indefinite blocking.
      if (turnId) {
        if (reflectionsSeenRef.current.has(turnId)) {
          setMessages(prev => [...prev, { role: 'assistant', content: assistantContent, timestamp: assistantTimestamp }]);
        } else {
          setPendingAnswer(turnId, { content: assistantContent, timestamp: assistantTimestamp });
          if (!STRICT_REFLECTION_FIRST) {
            const timeoutId = setTimeout(() => {
              if (pendingAnswersRef.current.has(turnId)) {
                const pending = pendingAnswersRef.current.get(turnId);
                pendingAnswersRef.current.delete(turnId);
                pendingTimeoutsRef.current.delete(turnId);
                setMessages(prev => [...prev, { role: 'assistant', content: pending?.content || assistantContent, timestamp: pending?.timestamp || assistantTimestamp }]);
              }
            }, 8000);
            pendingTimeoutsRef.current.set(turnId, timeoutId);
          }
        }
      } else {
        // No turn_id returned, show immediately
        setMessages(prev => [...prev, { role: 'assistant', content: assistantContent, timestamp: assistantTimestamp }]);
      }
    } catch (error) {
      logger.error('Submit error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="flex h-full w-full">
        {/* Chat panel (left) */}
        <div className="flex flex-col flex-1 h-full max-w-2xl min-w-0">
          <div className="flex-1 overflow-y-auto p-4 space-y-4 mb-4">
            {!apiBase ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500">
                  <p className="text-sm">Loading configuration...</p>
                </div>
              </div>
            ) : messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-500">
                  <p className="text-lg mb-2">Welcome to SELO DSP</p>
                  <p className="text-sm">Start a conversation with your SELO</p>
                </div>
              </div>
            ) : (
              messages.map((message, index) => {
                const displayTime = formatRelativeTime(message.timestamp);
                return (
                  <div
                    key={index}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[75%] rounded-lg p-4 ${
                        message.role === 'user'
                          ? 'bg-[var(--color-bg-elev-1)] border border-[var(--color-accent)]/40 text-white'
                          : 'bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] text-[var(--color-text-secondary)]'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      <div className="mt-2 text-xs text-[var(--color-text-muted)] text-right">{displayTime}</div>
                    </div>
                  </div>
                );
              })
            )}
            {isLoading && (
              <div className="flex items-center space-x-2 text-[var(--color-accent)]">
                <div className="w-2 h-2 bg-[var(--color-accent)] rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-[var(--color-accent)] rounded-full animate-pulse delay-150"></div>
                <div className="w-2 h-2 bg-[var(--color-accent)] rounded-full animate-pulse delay-300"></div>
                <span className="ml-2 text-sm">SELO DSP is thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          {/* Input area */}
          <form onSubmit={handleSubmit} className="mt-4">
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 bg-[var(--color-bg-elev-1)] border border-[var(--color-border)] rounded-lg px-4 py-2 text-white focus:outline-none"
                disabled={isLoading}
              />
              {messages.length > 0 && (
                <button
                  type="button"
                  onClick={clearConversation}
                  className="px-4 py-2 bg-[var(--color-bg-elev-1)] hover:bg-[var(--color-bg-elev-2)] text-gray-400 hover:text-white border border-[var(--color-border)] rounded-lg font-medium transition-colors"
                  title="Clear conversation"
                >
                  Clear
                </button>
              )}
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="px-6 py-2 bg-[var(--color-bg-elev-1)] hover:bg-[var(--color-bg-elev-2)] text-[var(--color-accent)] border border-[var(--color-accent)]/50 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Send
              </button>
            </div>
          </form>
        </div>
        {/* Reflection panel (right). Hidden on small screens to avoid overflow. */}
        <div className="hidden md:block md:w-[420px] md:min-w-[320px] md:max-w-[520px] h-full border-l border-[var(--color-border)] bg-[var(--color-bg-elev-1)]">
          <ErrorBoundary>
            <ReflectionPanel sessionId={sessionId} messages={messages} />
          </ErrorBoundary>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default Chat;
