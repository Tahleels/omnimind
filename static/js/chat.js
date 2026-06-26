/**
 * chat.js — Chat panel rendering: messages, typing indicator, sources panel.
 */

// ---- Formatting helpers ----

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatAnswer(text) {
  // Code blocks
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, _lang, code) =>
    `<pre><code>${escapeHtml(code.trim())}</code></pre>`
  );
  // Inline code
  text = text.replace(/`([^`]+)`/g,
    '<code style="font-family:JetBrains Mono,monospace;background:rgba(0,0,0,0.3);padding:1px 4px;border-radius:3px;font-size:0.82em">$1</code>'
  );
  // Bold
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Citations
  text = text.replace(/\[(\d+)\]/g, '<span class="cite">$1</span>');
  // Line breaks
  text = text.replace(/\n/g, '<br>');
  return text;
}

// ---- Welcome screen ----

export function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.remove();
}

// ---- Add a message bubble ----

export function addMessage(role, content, stats = null, route = null) {
  hideWelcome();
  const container = document.getElementById('messages');

  const msg = document.createElement('div');
  msg.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = role === 'user' ? '👤' : '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  if (role === 'ai' || role === 'assistant') {
    bubble.innerHTML = formatAnswer(content);

    if (stats || route) {
      const statsBar = document.createElement('div');
      statsBar.className = 'msg-stats';
      if (route) {
        statsBar.innerHTML += `<span class="route-chip ${route}">${route}</span>`;
      }
      if (stats && route === 'rag') {
        statsBar.innerHTML += `
          <span class="stat-chip">vec: ${stats.vector_results}</span>
          <span class="stat-chip">bm25: ${stats.bm25_results}</span>
          <span class="stat-chip">fused: ${stats.fused_results}</span>
          <span class="stat-chip">ranked: ${stats.final_chunks}</span>
          ${stats.invalid_citations_stripped > 0
            ? `<span class="stat-chip" style="color:#f87171">stripped: ${stats.invalid_citations_stripped}</span>`
            : ''}
        `;
      }
      bubble.appendChild(statsBar);
    }
  } else {
    bubble.textContent = content;
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

// ---- Typing indicator ----

export function addTyping() {
  hideWelcome();
  const container = document.getElementById('messages');
  const msg = document.createElement('div');
  msg.className = 'msg ai';
  msg.id = 'typing-msg';
  msg.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

export function removeTyping() {
  const t = document.getElementById('typing-msg');
  if (t) t.remove();
}

// ---- Sources panel ----

export function renderSources(sources, stats) {
  const content = document.getElementById('sources-content');
  const countEl = document.getElementById('sources-count');
  const statsPanel = document.getElementById('pipeline-stats');
  const statsGrid = document.getElementById('stats-grid');

  if (!sources || sources.length === 0) {
    content.innerHTML = '<div class="sources-empty">No sources cited.</div>';
    if (countEl) countEl.textContent = '';
    if (statsPanel) statsPanel.style.display = 'none';
    return;
  }

  if (countEl) countEl.textContent = `${sources.length} source${sources.length > 1 ? 's' : ''}`;
  content.innerHTML = '';

  sources.forEach(s => {
    const card = document.createElement('a');
    card.className = 'source-card';
    card.href = s.url;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.innerHTML = `
      <div class="source-num">${s.citation_number}</div>
      <div class="source-title">${escapeHtml(s.title)}</div>
      <div class="source-excerpt">${escapeHtml(s.excerpt)}</div>
      <div class="source-url">${new URL(s.url).pathname}</div>
    `;
    content.appendChild(card);
  });

  if (stats && statsPanel && statsGrid) {
    statsPanel.style.display = 'block';
    statsGrid.innerHTML = `
      <div class="stat-item"><span>Vector hits</span><span class="stat-val">${stats.vector_results}</span></div>
      <div class="stat-item"><span>BM25 hits</span><span class="stat-val">${stats.bm25_results}</span></div>
      <div class="stat-item"><span>After RRF</span><span class="stat-val">${stats.fused_results}</span></div>
      <div class="stat-item"><span>Reranked</span><span class="stat-val">${stats.final_chunks}</span></div>
    `;
  }
}

// ---- Toast notifications ----

export function showToast(message, type = 'info', duration = 3500) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
  toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ---- Auto-resize textarea ----

export function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
