/**
 * memory.js — Session ID persistence and history loading.
 *
 * Stores the current session_id in localStorage so the user's
 * conversation survives page refreshes.  On load, fetches the
 * last 10 messages from GET /session/{id} and re-renders them.
 */

import { getSession } from './api.js';

const SESSION_KEY = 'askmydocs_session_id';
const WORKSPACE_KEY = 'askmydocs_workspace_id';

// ---- Session ID ----

export function loadSessionId() {
  return localStorage.getItem(SESSION_KEY) || null;
}

export function saveSessionId(id) {
  if (id) localStorage.setItem(SESSION_KEY, id);
}

export function clearSessionId() {
  localStorage.removeItem(SESSION_KEY);
}

// ---- Workspace preference ----

export function loadWorkspaceId() {
  return localStorage.getItem(WORKSPACE_KEY) || 'default';
}

export function saveWorkspaceId(id) {
  if (id) localStorage.setItem(WORKSPACE_KEY, id);
}

/**
 * Attempt to restore conversation history from the server.
 *
 * @param {string} sessionId
 * @param {function} onMessage  Called with (role, content) for each message.
 * @returns {Promise<boolean>}  True if history was found and rendered.
 */
export async function restoreHistory(sessionId, onMessage) {
  if (!sessionId) return false;

  try {
    const session = await getSession(sessionId);
    if (!session || !session.messages || session.messages.length === 0) {
      return false;
    }

    for (const msg of session.messages) {
      onMessage(msg.role, msg.content);
    }
    return true;
  } catch {
    // Session expired or server restarted — clear stale ID
    clearSessionId();
    return false;
  }
}
