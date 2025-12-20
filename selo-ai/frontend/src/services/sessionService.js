// Session service to centralize creation/retrieval of a persistent session/user ID
// Ensures a single, stable identifier is used across the app for Socket.IO and API calls.

import { v4 as uuidv4 } from 'uuid';

const SESSION_STORAGE_KEY = 'selo_ai_session_id';

export function getOrCreateSessionId() {
  try {
    let sid = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!sid || typeof sid !== 'string' || sid.trim() === '') {
      sid = uuidv4();
      localStorage.setItem(SESSION_STORAGE_KEY, sid);
    }
    return sid;
  } catch (_e) {
    // Fallback if localStorage is unavailable (very rare in our context)
    return uuidv4();
  }
}

export function getSessionId() {
  try {
    return localStorage.getItem(SESSION_STORAGE_KEY) || null;
  } catch (_e) {
    return null;
  }
}

export function setSessionId(id) {
  if (!id) return;
  try {
    localStorage.setItem(SESSION_STORAGE_KEY, id);
  } catch (_e) {}
}
