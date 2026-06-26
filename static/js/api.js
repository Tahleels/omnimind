/**
 * api.js — Fetch wrappers for all v2 API endpoints.
 *
 * All functions return parsed JSON or throw an Error with a user-friendly
 * message.  The UI layer calls these and handles loading/error states.
 */

const BASE = '';   // Same-origin — no need to set a base URL

export async function checkHealth() {
  const r = await fetch(`${BASE}/health`);
  if (!r.ok) throw new Error('Server unreachable');
  return r.json();
}

/**
 * POST /ask
 * @param {string} question
 * @param {string|null} sessionId
 * @param {string} workspaceId
 * @param {string} searchMode  "docs" | "workspace" | "combined"
 */
export async function askQuestion(question, sessionId, workspaceId = 'default', searchMode = 'combined') {
  const body = { question, workspace_id: workspaceId, search_mode: searchMode };
  if (sessionId) body.session_id = sessionId;

  const r = await fetch(`${BASE}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${r.status}`);
  }
  return r.json();
}

/**
 * GET /session/{id}
 */
export async function getSession(sessionId) {
  const r = await fetch(`${BASE}/session/${sessionId}`);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Failed to load session: ${r.status}`);
  return r.json();
}

/**
 * DELETE /session/{id}
 */
export async function clearSession(sessionId) {
  const r = await fetch(`${BASE}/session/${sessionId}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`Failed to clear session: ${r.status}`);
  return r.json();
}

/**
 * GET /workspaces
 */
export async function listWorkspaces() {
  const r = await fetch(`${BASE}/workspaces`);
  if (!r.ok) throw new Error('Failed to list workspaces');
  return r.json();
}

/**
 * POST /workspaces
 */
export async function createWorkspace(name, description = '') {
  const r = await fetch(`${BASE}/workspaces`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to create workspace');
  }
  return r.json();
}

/**
 * DELETE /workspaces/{id}
 */
export async function deleteWorkspace(workspaceId) {
  const r = await fetch(`${BASE}/workspaces/${workspaceId}`, { method: 'DELETE' });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete workspace');
  }
  return r.json();
}

/**
 * POST /workspaces/{id}/ingest/url
 */
export async function ingestUrl(workspaceId, url, name = null) {
  const body = { url };
  if (name) body.name = name;
  const r = await fetch(`${BASE}/workspaces/${workspaceId}/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || 'URL ingestion failed');
  }
  return r.json();
}

/**
 * POST /workspaces/{id}/ingest/file  (multipart/form-data)
 */
export async function ingestFile(workspaceId, file) {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${BASE}/workspaces/${workspaceId}/ingest/file`, {
    method: 'POST',
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || 'File ingestion failed');
  }
  return r.json();
}
