/**
 * workspace.js — Sidebar workspace list, creation modal, and source display.
 */

import { listWorkspaces, createWorkspace, deleteWorkspace, ingestUrl, ingestFile } from './api.js';
import { showToast } from './chat.js';
import { saveWorkspaceId } from './memory.js';

let _workspaces = [];
let _activeWorkspaceId = 'default';
let _onWorkspaceChange = null;   // Callback: (workspaceId) => void

export function onWorkspaceChange(cb) { _onWorkspaceChange = cb; }
export function getActiveWorkspaceId() { return _activeWorkspaceId; }

// ---- Load & render workspaces ----

export async function loadWorkspaces(initialId = 'default') {
  try {
    const data = await listWorkspaces();
    _workspaces = data.workspaces || [];
    _activeWorkspaceId = initialId;
    renderWorkspaceList();
  } catch (e) {
    console.error('Failed to load workspaces:', e);
  }
}

function renderWorkspaceList() {
  const list = document.getElementById('workspace-list');
  if (!list) return;
  list.innerHTML = '';

  _workspaces.forEach(ws => {
    // Workspace row
    const item = document.createElement('div');
    item.className = `ws-item${ws.workspace_id === _activeWorkspaceId ? ' active' : ''}`;
    item.id = `ws-${ws.workspace_id}`;
    const icon = ws.is_default ? '📚' : '📂';
    item.innerHTML = `
      <span class="ws-icon">${icon}</span>
      <span class="ws-name">${ws.name}</span>
      <span class="ws-count">${ws.total_chunks > 0 ? ws.total_chunks + ' chunks' : '—'}</span>
    `;
    item.addEventListener('click', () => setActiveWorkspace(ws.workspace_id));
    list.appendChild(item);

    // Sources sub-list (only show if workspace has sources)
    if (ws.sources && ws.sources.length > 0) {
      const sources = document.createElement('div');
      sources.className = 'ws-sources';
      ws.sources.forEach(src => {
        const srcItem = document.createElement('div');
        srcItem.className = 'ws-source-item';
        const typeIcon = { pdf: '📄', docx: '📝', url: '🌐', youtube: '▶️', txt: '📋', langchain_docs: '📚' }[src.source_type] || '📎';
        srcItem.innerHTML = `<span class="ws-source-type">${typeIcon}</span><span>${src.name}</span>`;
        sources.appendChild(srcItem);
      });
      list.appendChild(sources);
    }
  });
}

function setActiveWorkspace(wsId) {
  _activeWorkspaceId = wsId;
  saveWorkspaceId(wsId);
  renderWorkspaceList();
  if (_onWorkspaceChange) _onWorkspaceChange(wsId);
}

// ---- New workspace modal ----

export function openNewWorkspaceModal() {
  document.getElementById('modal-overlay').classList.add('show');
  document.getElementById('ws-name-input').value = '';
  document.getElementById('ws-desc-input').value = '';
  document.getElementById('ws-name-input').focus();
}

export function closeModal() {
  document.getElementById('modal-overlay').classList.remove('show');
}

export async function submitNewWorkspace() {
  const name = document.getElementById('ws-name-input').value.trim();
  if (!name) { showToast('Please enter a workspace name.', 'error'); return; }

  try {
    const ws = await createWorkspace(name, document.getElementById('ws-desc-input').value.trim());
    showToast(`Workspace "${ws.name}" created!`, 'success');
    closeModal();
    await loadWorkspaces(_activeWorkspaceId);
    setActiveWorkspace(ws.workspace_id);
  } catch (e) {
    showToast(e.message, 'error');
  }
}

// ---- File/URL ingestion with progress overlay ----

const STEPS = ['Parsing document…', 'Generating embeddings…', 'Indexing into workspace…'];

function setUploadStep(stepIdx, done = false) {
  document.querySelectorAll('.upload-step').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i < stepIdx) el.classList.add('done');
    else if (i === stepIdx) el.classList.add(done ? 'done' : 'active');
  });
  const pct = done ? 100 : Math.round(((stepIdx) / STEPS.length) * 100);
  document.querySelector('.upload-progress-fill').style.width = pct + '%';
}

function showUploadOverlay(filename) {
  const overlay = document.getElementById('upload-overlay');
  document.getElementById('upload-filename').textContent = filename;
  document.querySelectorAll('.upload-step').forEach(el => el.classList.remove('active', 'done'));
  document.querySelector('.upload-progress-fill').style.width = '0%';
  overlay.classList.add('show');
}

function hideUploadOverlay() {
  document.getElementById('upload-overlay').classList.remove('show');
}

export async function handleFileUpload(file) {
  if (!file) return;

  const ws = _workspaces.find(w => w.workspace_id === _activeWorkspaceId);
  if (!ws || ws.is_default) {
    showToast('Please select or create a custom workspace first.', 'error');
    return;
  }

  showUploadOverlay(file.name);

  try {
    setUploadStep(0);
    await new Promise(r => setTimeout(r, 400));  // Let UI update
    setUploadStep(1);
    const result = await ingestFile(_activeWorkspaceId, file);
    setUploadStep(2);
    await new Promise(r => setTimeout(r, 300));
    setUploadStep(2, true);
    await new Promise(r => setTimeout(r, 400));
    hideUploadOverlay();
    showToast(`✓ Indexed "${result.name}" (${result.chunks_indexed} chunks)`, 'success');
    await loadWorkspaces(_activeWorkspaceId);
  } catch (e) {
    hideUploadOverlay();
    showToast(e.message, 'error');
  }
}

export async function handleUrlIngest(url) {
  if (!url) return;

  const ws = _workspaces.find(w => w.workspace_id === _activeWorkspaceId);
  if (!ws || ws.is_default) {
    showToast('Please select or create a custom workspace first.', 'error');
    return;
  }

  showUploadOverlay(url.length > 50 ? url.slice(0, 50) + '…' : url);

  try {
    setUploadStep(0);
    await new Promise(r => setTimeout(r, 200));
    setUploadStep(1);
    const result = await ingestUrl(_activeWorkspaceId, url);
    setUploadStep(2);
    await new Promise(r => setTimeout(r, 300));
    setUploadStep(2, true);
    await new Promise(r => setTimeout(r, 400));
    hideUploadOverlay();
    showToast(`✓ Indexed "${result.name}" (${result.chunks_indexed} chunks)`, 'success');
    await loadWorkspaces(_activeWorkspaceId);
  } catch (e) {
    hideUploadOverlay();
    showToast(e.message, 'error');
  }
}
