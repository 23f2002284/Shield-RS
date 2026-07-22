/**
 * Shield v3 — YouTube-Clone Frontend
 * ====================================
 * Features:
 *   - Auto-save watch history when video plays (YouTube IFrame API)
 *   - Stop video on page navigation
 *   - Dynamic topic suggestions in search
 *   - Dark theme default
 *   - Full SPA: auth, feed, search, compare, watch, settings
 */

const API = '/api';

// ── State ──
let currentUser = null;
let feedCache = [];
let feedTopics = [];
let currentPage = 'home';
let currentVideo = null;
let ytPlayer = null;           // YouTube IFrame API player
let watchStartTime = null;     // When video started playing
let watchAutoSaved = false;    // Prevent double-saving

// ── Dynamic Topic Suggestions ──
const TOPIC_SUGGESTIONS = [
  { text: 'Machine Learning tutorial', icon: 'search' },
  { text: 'Climate change documentary', icon: 'search' },
  { text: 'How to invest for beginners', icon: 'search' },
  { text: 'Ancient history documentary', icon: 'search' },
  { text: 'Healthy meal prep ideas', icon: 'search' },
  { text: 'Home workout routine', icon: 'search' },
  { text: 'Psychology explained', icon: 'search' },
  { text: 'Space exploration documentary', icon: 'search' },
  { text: 'Web development crash course', icon: 'search' },
  { text: 'Quantum physics for beginners', icon: 'search' },
  { text: 'Music theory basics', icon: 'search' },
  { text: 'Filmmaking tips', icon: 'search' },
  { text: 'Startup advice', icon: 'search' },
  { text: 'Learn a new language', icon: 'search' },
  { text: 'Philosophy of life', icon: 'search' },
  { text: 'Biology explained', icon: 'search' },
  { text: 'Game design tutorial', icon: 'search' },
  { text: 'World politics explained', icon: 'search' },
  { text: 'Digital art for beginners', icon: 'search' },
  { text: 'DIY engineering projects', icon: 'search' },
];

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  loadYouTubeAPI();
  restoreSession();
  setupNav();
  setupAuth();
  setupSearch();
  setupCompare();
  setupSettings();
  loadFeed().then(() => {
    // Handle URL routing after feed is loaded (so we have cache for watch etc)
    handleUrlRoute();
  });
});

window.addEventListener('popstate', (e) => {
  if (e.state && e.state.page) {
    if (e.state.page === 'watch' && e.state.videoId) {
      // Find video and open it without pushing state again
      const video = feedCache.find(v => v.video_id === e.state.videoId);
      if (video) {
        openWatch(video, false);
      } else {
        // Fallback fetch
        fetch(`${API}/videos/${e.state.videoId}`)
          .then(r => r.json())
          .then(v => {
            if (v && v.video_id) openWatch(v, false);
            else navigateTo('home', false);
          }).catch(() => navigateTo('home', false));
      }
    } else {
      navigateTo(e.state.page, false);
    }
  } else {
    // Fallback to URL parsing
    handleUrlRoute(false);
  }
});

async function handleUrlRoute(pushStateFlag = true) {
  const path = window.location.pathname;
  if (path.startsWith('/watch')) {
    const params = new URLSearchParams(window.location.search);
    const vId = params.get('v') || path.split('/').pop();
    if (vId && vId !== 'watch') {
      try {
        const res = await fetch(`${API}/videos/${vId}`);
        if (res.ok) {
          const video = await res.json();
          openWatch(video, pushStateFlag);
          return;
        }
      } catch (e) {
        console.error('Failed to fetch video for routing:', e);
      }
    }
    navigateTo('home', pushStateFlag);
  } else if (path === '/trending') {
    navigateTo('trending', pushStateFlag);
  } else if (path === '/history') {
    navigateTo('history', pushStateFlag);
  } else if (path === '/settings') {
    navigateTo('settings', pushStateFlag);
  } else if (path === '/compare') {
    navigateTo('compare', pushStateFlag);
  } else {
    navigateTo('home', pushStateFlag);
  }
}

let watchUpdateInterval = null;

// ============================================================
// YouTube IFrame API (for auto-watch tracking)
// ============================================================
function loadYouTubeAPI() {
  const tag = document.createElement('script');
  tag.src = 'https://www.youtube.com/iframe_api';
  document.head.appendChild(tag);
}

// Global callback required by YouTube IFrame API
window.onYouTubeIframeAPIReady = function() {
  console.log('[Shield] YouTube IFrame API ready');
};

function createYTPlayer(videoId, startSeconds = 0) {
  destroyYTPlayer();
  watchStartTime = null;

  const container = document.getElementById('watch-player');
  container.innerHTML = '<div id="yt-player-target"></div>';

  if (typeof YT !== 'undefined' && YT.Player) {
    ytPlayer = new YT.Player('yt-player-target', {
      videoId: videoId,
      width: '100%',
      height: '100%',
      playerVars: {
        autoplay: 1,
        modestbranding: 1,
        rel: 0,
        start: startSeconds
      },
      events: {
        onStateChange: onPlayerStateChange,
      },
    });
  } else {
    // Fallback: plain iframe if API not loaded yet
    container.innerHTML = `
      <iframe src="https://www.youtube.com/embed/${videoId}?autoplay=1&modestbranding=1&rel=0&start=${startSeconds}"
        allow="autoplay; encrypted-media" allowfullscreen
        style="width:100%;height:100%;border:none;"></iframe>
    `;
    // If fallback, we just save immediately at 0% since we can't track
    setTimeout(() => saveCurrentProgress(false), 1000);
  }
}

function onPlayerStateChange(event) {
  // YT.PlayerState: PLAYING=1, PAUSED=2, ENDED=0
  if (event.data === 1) {
    if (!watchStartTime) {
      // Video started playing for the first time
      watchStartTime = Date.now();
      saveCurrentProgress(true); // show toast on first start
    }
    
    // Start periodic saving every 10 seconds
    if (!watchUpdateInterval) {
      watchUpdateInterval = setInterval(() => {
        saveCurrentProgress(false);
      }, 10000);
    }
  } else {
    // PAUSED or ENDED
    if (watchUpdateInterval) {
      clearInterval(watchUpdateInterval);
      watchUpdateInterval = null;
    }
    
    if (event.data === 0 || event.data === 2) {
      saveCurrentProgress(event.data === 0); // show toast if ended
    }
    if (event.data === 0) {
      watchStartTime = null; // reset if ended
    }
  }
}

function saveCurrentProgress(showToastFlag = false) {
  if (!watchStartTime || !currentVideo || !currentUser) return;
  const watchedSeconds = Math.round((Date.now() - watchStartTime) / 1000);
  const totalSeconds = currentVideo.duration_seconds || 300;
  const watchPct = Math.min(1.0, watchedSeconds / Math.max(totalSeconds, 1));
  
  autoSaveWatch(currentVideo, watchPct, watchedSeconds, showToastFlag);
}

function destroyYTPlayer() {
  if (watchUpdateInterval) {
    clearInterval(watchUpdateInterval);
    watchUpdateInterval = null;
  }
  
  // Save final progress
  saveCurrentProgress(false);
  
  if (ytPlayer && typeof ytPlayer.destroy === 'function') {
    try { ytPlayer.destroy(); } catch {}
  }
  ytPlayer = null;
  watchStartTime = null;

  // Also clear any plain iframes
  const container = document.getElementById('watch-player');
  if (container) container.innerHTML = '';
}

async function autoSaveWatch(video, watchPct, watchedSeconds, showToastFlag = false) {
  if (!currentUser) return;

  try {
    await fetch(`${API}/users/${currentUser.id}/history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: video.video_id || '',
        title: video.title || '',
        description: video.description || '',
        channel: video.channel_name || video.channel || '',
        thumbnail: video.thumbnail_url || video.thumbnail || '',
        duration_seconds: video.duration_seconds || 0,
        view_count: video.view_count || 0,
        like_count: video.like_count || 0,
        subscriber_count: video.subscriber_count || 0,
        watch_pct: watchPct,
        agent_scores: video.agent_scores || {},
      }),
    });

    if (showToastFlag) {
      showToast(`Saved to history (${formatDuration(watchedSeconds)} watched)`);
    }
  } catch (err) {
    console.error('Auto-save watch failed:', err);
  }
}

function showToast(message) {
  const existing = document.querySelector('.auto-watch-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'auto-watch-toast';
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ============================================================
// Session Management
// ============================================================
function restoreSession() {
  const saved = localStorage.getItem('shield_user');
  if (saved) {
    try {
      currentUser = JSON.parse(saved);
      updateAuthUI();
      // Asynchronously verify that the user still exists in the backend
      verifySession(currentUser.id);
    } catch { currentUser = null; }
  }
}

async function verifySession(userId) {
  try {
    const res = await fetch(`${API}/users/${userId}`);
    if (!res.ok && res.status === 404) {
      console.warn('Cached user not found on backend. Clearing session.');
      clearSession();
      showToast('Session expired. Please sign in again.');
      
      // If we are currently on a page that requires auth, redirect home or refresh UI
      if (['history', 'settings'].includes(currentPage)) {
        navigateTo('home');
      }
    }
  } catch (err) {
    console.error('Failed to verify session:', err);
  }
}

function saveSession(user) {
  currentUser = user;
  localStorage.setItem('shield_user', JSON.stringify(user));
  updateAuthUI();
}

function clearSession() {
  currentUser = null;
  localStorage.removeItem('shield_user');
  updateAuthUI();
}

function updateAuthUI() {
  const signinBtn = document.getElementById('signin-btn');
  const userMenu = document.getElementById('user-menu');

  if (currentUser) {
    signinBtn.classList.add('hidden');
    userMenu.classList.remove('hidden');
    const initial = (currentUser.name || '?')[0].toUpperCase();
    document.getElementById('avatar-btn').textContent = initial;
    document.getElementById('dropdown-avatar').textContent = initial;
    document.getElementById('dropdown-name').textContent = currentUser.name;
    document.getElementById('dropdown-email').textContent = currentUser.email || '';
  } else {
    signinBtn.classList.remove('hidden');
    userMenu.classList.add('hidden');
  }
}

// ============================================================
// Navigation (stops video on page change)
// ============================================================
function setupNav() {
  document.querySelectorAll('.sidebar-item[data-page]').forEach(btn => {
    btn.addEventListener('click', () => navigateTo(btn.dataset.page));
  });

  document.getElementById('menu-toggle').addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('collapsed');
  });

  document.getElementById('logo-home').addEventListener('click', (e) => {
    e.preventDefault();
    navigateTo('home');
  });

  document.querySelectorAll('.dropdown-item[data-page]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('user-dropdown').classList.add('hidden');
      navigateTo(btn.dataset.page);
    });
  });

  document.getElementById('avatar-btn')?.addEventListener('click', () => {
    document.getElementById('user-dropdown').classList.toggle('hidden');
  });

  document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('user-dropdown');
    const menu = document.getElementById('user-menu');
    if (dropdown && !dropdown.classList.contains('hidden') && !menu.contains(e.target)) {
      dropdown.classList.add('hidden');
    }
  });

  document.getElementById('signout-btn')?.addEventListener('click', () => {
    clearSession();
    document.getElementById('user-dropdown').classList.add('hidden');
    navigateTo('home');
  });
}

function navigateTo(page, pushStateFlag = true) {
  // ── STOP VIDEO when leaving watch page ──
  if (currentPage === 'watch' && page !== 'watch') {
    destroyYTPlayer();
  }

  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.sidebar-item').forEach(b => b.classList.remove('active'));

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add('active');

  const navEl = document.getElementById(`nav-${page}`);
  if (navEl) navEl.classList.add('active');

  currentPage = page;

  if (pushStateFlag) {
    if (page === 'home') {
      history.pushState({ page: 'home' }, '', '/');
    } else if (page !== 'watch') {
      history.pushState({ page }, '', `/${page}`);
    }
  }

  if (page === 'trending') loadTrending();
  if (page === 'history') loadHistory();
  if (page === 'settings') loadSettings();
}

// ============================================================
// Auth
// ============================================================
function setupAuth() {
  const modal = document.getElementById('auth-modal');
  const loginForm = document.getElementById('login-form');
  const registerForm = document.getElementById('register-form');

  document.getElementById('signin-btn').addEventListener('click', () => {
    modal.classList.remove('hidden');
    loginForm.classList.remove('hidden');
    registerForm.classList.add('hidden');
  });

  document.getElementById('auth-close').addEventListener('click', () => {
    modal.classList.add('hidden');
  });
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.add('hidden');
  });

  document.getElementById('show-register').addEventListener('click', (e) => {
    e.preventDefault();
    loginForm.classList.add('hidden');
    registerForm.classList.remove('hidden');
  });
  document.getElementById('show-login').addEventListener('click', (e) => {
    e.preventDefault();
    registerForm.classList.add('hidden');
    loginForm.classList.remove('hidden');
  });

  // Login
  document.getElementById('login-btn').addEventListener('click', async () => {
    const email = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('login-error');

    if (!email || !password) {
      errorEl.textContent = 'Please enter email and password';
      errorEl.classList.remove('hidden');
      return;
    }

    try {
      const res = await fetch(`${API}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        errorEl.textContent = data.detail || 'Login failed';
        errorEl.classList.remove('hidden');
        return;
      }
      saveSession(data);
      modal.classList.add('hidden');
      loadFeed();
    } catch (err) {
      errorEl.textContent = 'Connection error';
      errorEl.classList.remove('hidden');
    }
  });

  // Register
  document.getElementById('register-btn').addEventListener('click', async () => {
    const name = document.getElementById('reg-name').value.trim();
    const email = document.getElementById('reg-email').value.trim();
    const password = document.getElementById('reg-password').value;
    const location = document.getElementById('reg-location').value;
    const language = document.getElementById('reg-language').value;
    const strictness = document.getElementById('reg-strictness').value;
    const errorEl = document.getElementById('register-error');

    const topics = [];
    document.querySelectorAll('#reg-topics input:checked').forEach(cb => {
      topics.push(cb.value);
    });

    if (!name || !email || !password) {
      errorEl.textContent = 'Please fill in name, email, and password';
      errorEl.classList.remove('hidden');
      return;
    }
    if (password.length < 6) {
      errorEl.textContent = 'Password must be at least 6 characters';
      errorEl.classList.remove('hidden');
      return;
    }

    try {
      const res = await fetch(`${API}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name, email, password, location, language,
          preferred_topics: topics,
          content_strictness: strictness,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        errorEl.textContent = data.detail || 'Registration failed';
        errorEl.classList.remove('hidden');
        return;
      }
      saveSession(data);
      modal.classList.add('hidden');
      loadFeed();
    } catch (err) {
      errorEl.textContent = 'Connection error';
      errorEl.classList.remove('hidden');
    }
  });
}

// ============================================================
// Feed (Home Page)
// ============================================================
async function loadFeed() {
  const grid = document.getElementById('feed-grid');
  const filterBar = document.getElementById('topic-filter-bar');

  try {
    let url = `${API}/feed?limit=2000`;
    if (currentUser) {
      url += `&user_id=${currentUser.id}`;
    }

    const res = await fetch(url);
    const data = await res.json();

    feedCache = data.videos || [];
    feedTopics = data.topics || [];

    // Build topic filter chips
    filterBar.innerHTML = '<button class="filter-chip active" data-topic="all">All</button>';
    feedTopics.forEach(topic => {
      const btn = document.createElement('button');
      btn.className = 'filter-chip';
      btn.dataset.topic = topic;
      btn.textContent = topic;
      filterBar.appendChild(btn);
    });

    filterBar.querySelectorAll('.filter-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        filterBar.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        renderFeed(chip.dataset.topic);
      });
    });

    // Populate sidebar topics
    const sidebarTopics = document.getElementById('sidebar-topics');
    sidebarTopics.innerHTML = '';
    feedTopics.forEach(topic => {
      const btn = document.createElement('button');
      btn.className = 'sidebar-topic-btn';
      btn.textContent = topic;
      btn.addEventListener('click', () => {
        filterBar.querySelectorAll('.filter-chip').forEach(c => {
          c.classList.toggle('active', c.dataset.topic === topic);
        });
        navigateTo('home');
        renderFeed(topic);
      });
      sidebarTopics.appendChild(btn);
    });

    renderFeed('all');
  } catch (err) {
    console.error('Feed load failed:', err);
    grid.innerHTML = '<div class="empty-state"><p>Failed to load feed. Is the backend running?</p></div>';
  }
}

function renderFeed(topic) {
  const grid = document.getElementById('feed-grid');
  let videos = feedCache;

  if (topic !== 'all') {
    videos = videos.filter(v => v.topic === topic);
  }

  grid.innerHTML = '';
  videos.forEach(v => {
    grid.appendChild(createVideoCard(v));
  });

  if (videos.length === 0) {
    grid.innerHTML = '<div class="empty-state"><p>No videos in this category yet.</p></div>';
  }
}

// ============================================================
// Search with Dynamic Suggestions
// ============================================================
function setupSearch() {
  const input = document.getElementById('global-search');
  const btn = document.getElementById('global-search-btn');
  const suggestionsEl = document.getElementById('search-suggestions');

  const doSearch = () => {
    const q = input.value.trim();
    if (!q) return;
    suggestionsEl.classList.add('hidden');
    navigateTo('search');
    performSearch(q);
  };

  btn.addEventListener('click', doSearch);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch();
    if (e.key === 'Escape') suggestionsEl.classList.add('hidden');
  });

  // Advanced Options UI Logic
  const filterBtn = document.getElementById('global-search-filter-btn');
  const advPanel = document.getElementById('advanced-search-panel');
  const maxInput = document.getElementById('adv-search-max');
  const maxSpan = document.getElementById('adv-max-val');
  const timeInput = document.getElementById('adv-search-time');
  const timeSpan = document.getElementById('adv-time-val');

  filterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    // Hide suggestions when toggling options
    suggestionsEl.classList.add('hidden');
    
    const isHidden = advPanel.classList.toggle('hidden');
    if (!isHidden) {
      filterBtn.classList.add('active');
    } else {
      filterBtn.classList.remove('active');
    }
  });

  maxInput.addEventListener('input', () => maxSpan.textContent = maxInput.value);
  timeInput.addEventListener('input', () => timeSpan.textContent = timeInput.value);

  // Dynamic suggestions on focus/input
  input.addEventListener('focus', () => showSuggestions(input.value));
  input.addEventListener('input', () => showSuggestions(input.value));

  // Hide on click outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.topbar-search-wrapper')) {
      suggestionsEl.classList.add('hidden');
      advPanel.classList.add('hidden');
      filterBtn.classList.remove('active');
    }
  });
}

function showSuggestions(query) {
  const suggestionsEl = document.getElementById('search-suggestions');
  const q = query.toLowerCase().trim();

  let items = [];

  // Section 1: Matching catalog topics
  if (q.length > 0) {
    const matching = TOPIC_SUGGESTIONS.filter(s =>
      s.text.toLowerCase().includes(q)
    ).slice(0, 5);
    if (matching.length > 0) items.push(...matching);

    // Section 2: Search within cached video titles
    const titleMatches = feedCache
      .filter(v => v.title && v.title.toLowerCase().includes(q))
      .slice(0, 3)
      .map(v => ({ text: v.title.substring(0, 60), icon: 'video', video: v }));
    if (titleMatches.length > 0) items.push(...titleMatches);
  } else {
    // Show trending suggestions when empty
    const shuffled = [...TOPIC_SUGGESTIONS].sort(() => Math.random() - 0.5).slice(0, 8);
    items = shuffled;
  }

  if (items.length === 0) {
    suggestionsEl.classList.add('hidden');
    return;
  }

  suggestionsEl.innerHTML = '';

  // Add section header
  if (q.length === 0) {
    suggestionsEl.innerHTML = '<div class="suggestion-section">Trending searches</div>';
  }

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'suggestion-item';
    const iconSvg = item.icon === 'video'
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>';

    div.innerHTML = `${iconSvg}<span>${escapeHtml(item.text)}</span>`;
    div.addEventListener('click', () => {
      const input = document.getElementById('global-search');
      if (item.video) {
        // Direct to video
        openWatch(item.video);
      } else {
        input.value = item.text;
        suggestionsEl.classList.add('hidden');
        navigateTo('search');
        performSearch(item.text);
      }
    });
    suggestionsEl.appendChild(div);
  });

  suggestionsEl.classList.remove('hidden');
  
  // Hide advanced panel if showing suggestions
  const advPanel = document.getElementById('advanced-search-panel');
  const filterBtn = document.getElementById('global-search-filter-btn');
  if (advPanel && !advPanel.classList.contains('hidden')) {
    advPanel.classList.add('hidden');
    if (filterBtn) filterBtn.classList.remove('active');
  }
}

async function performSearch(query) {
  const grid = document.getElementById('search-grid');
  const header = document.getElementById('search-header');
  const metrics = document.getElementById('search-metrics');
  const loading = document.getElementById('search-loading');
  const empty = document.getElementById('search-empty');

  grid.innerHTML = '';
  header.innerHTML = '';
  metrics.innerHTML = '';
  loading.classList.remove('hidden');
  empty.classList.add('hidden');

  // Sync to compare input
  const compareInput = document.getElementById('compare-input');
  if (compareInput) compareInput.value = query;

  // Read advanced constraints
  const maxVideos = parseInt(document.getElementById('adv-search-max')?.value || 20, 10);
  const timeBudget = parseInt(document.getElementById('adv-search-time')?.value || 60, 10);
  const qualityPref = document.getElementById('adv-search-pref')?.value || 'balanced';

  // Hide advanced panel if open
  document.getElementById('advanced-search-panel')?.classList.add('hidden');
  document.getElementById('global-search-filter-btn')?.classList.remove('active');

  try {
    const res = await fetch(`${API}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        max_results: maxVideos,
        time_budget_minutes: timeBudget,
        quality_preference: qualityPref
      }),
    });
    const data = await res.json();

    loading.classList.add('hidden');

    const videos = data.videos || [];
    if (videos.length === 0) {
      empty.classList.remove('hidden');
      return;
    }

    header.innerHTML = `Showing ${videos.length} results for "<strong>${escapeHtml(query)}</strong>" &middot; ${data.elapsed_ms}ms`;

    const m = data.metrics || {};
    metrics.innerHTML = `
      <div><span class="compare-metric-label">Avg Shield</span><br><span class="compare-metric-value">${pct(m.avg_shield_score)}</span></div>
      <div><span class="compare-metric-label">Avg Clickbait</span><br><span class="compare-metric-value">${pct(m.avg_clickbait)}</span></div>
      <div><span class="compare-metric-label">Avg Credibility</span><br><span class="compare-metric-value">${pct(m.avg_credibility)}</span></div>
      <div><span class="compare-metric-label">Total Duration</span><br><span class="compare-metric-value">${m.total_duration_minutes || 0} min</span></div>
    `;

    videos.forEach(v => grid.appendChild(createVideoCard(v)));
  } catch (err) {
    loading.classList.add('hidden');
    grid.innerHTML = '<div class="empty-state"><p>Search failed. Check backend connection.</p></div>';
  }
}

// ============================================================
// Compare
// ============================================================
function setupCompare() {
  const input = document.getElementById('compare-input');
  const btn = document.getElementById('compare-btn');

  const doCompare = () => {
    const q = input.value.trim();
    if (!q) return;
    performCompare(q);
  };

  btn.addEventListener('click', doCompare);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doCompare();
  });
}

async function performCompare(query) {
  const container = document.getElementById('compare-container');
  const loading = document.getElementById('compare-loading');
  const ytResults = document.getElementById('youtube-results');
  const shResults = document.getElementById('shield-results');
  const ytMetrics = document.getElementById('yt-compare-metrics');
  const shMetrics = document.getElementById('sh-compare-metrics');

  container.classList.add('hidden');
  loading.classList.remove('hidden');

  try {
    const res = await fetch(`${API}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_results: 15 }),
    });
    const data = await res.json();

    loading.classList.add('hidden');
    container.classList.remove('hidden');

    const ytM = data.youtube?.metrics || {};
    ytMetrics.innerHTML = `
      <div><span class="compare-metric-label">Avg Shield</span><br><span class="compare-metric-value">${pct(ytM.avg_shield_score)}</span></div>
      <div><span class="compare-metric-label">Avg Clickbait</span><br><span class="compare-metric-value">${pct(ytM.avg_clickbait)}</span></div>
      <div><span class="compare-metric-label">Avg Credibility</span><br><span class="compare-metric-value">${pct(ytM.avg_credibility)}</span></div>
    `;

    const shM = data.shield?.metrics || {};
    shMetrics.innerHTML = `
      <div><span class="compare-metric-label">Avg Shield</span><br><span class="compare-metric-value">${pct(shM.avg_shield_score)}</span></div>
      <div><span class="compare-metric-label">Avg Clickbait</span><br><span class="compare-metric-value">${pct(shM.avg_clickbait)}</span></div>
      <div><span class="compare-metric-label">Avg Credibility</span><br><span class="compare-metric-value">${pct(shM.avg_credibility)}</span></div>
    `;

    ytResults.innerHTML = '';
    (data.youtube?.videos || []).forEach((v, i) => {
      ytResults.appendChild(createCompareItem(v, i + 1, 'youtube'));
    });

    shResults.innerHTML = '';
    (data.shield?.videos || []).forEach((v, i) => {
      shResults.appendChild(createCompareItem(v, i + 1, 'shield'));
    });
  } catch (err) {
    loading.classList.add('hidden');
    container.classList.remove('hidden');
    ytResults.innerHTML = '<div class="empty-state"><p>Compare failed.</p></div>';
  }
}

function createCompareItem(video, rank, side) {
  const div = document.createElement('div');
  div.className = 'compare-item';
  div.addEventListener('click', () => openWatch(video));

  const scores = video.agent_scores || {};
  const thumb = video.thumbnail_url || video.thumbnail || `https://img.youtube.com/vi/${video.video_id}/mqdefault.jpg`;

  let tagsHtml = '';
  if (side === 'youtube') {
    tagsHtml = `
      <span class="score-tag yt">${formatCount(video.view_count || 0)} views</span>
      <span class="score-tag yt">${formatCount(video.like_count || 0)} likes</span>
      <span class="score-tag yt">${formatCount(video.subscriber_count || 0)} subs</span>
    `;
  } else {
    tagsHtml = `
      <span class="score-tag ${scoreColor(scores.shield_score)}">Shield ${pct(scores.shield_score)}</span>
      <span class="score-tag ${scoreColor(scores.credibility)}">Cred ${pct(scores.credibility)}</span>
      <span class="score-tag ${cbColor(scores.clickbait)}">CB ${pct(scores.clickbait)}</span>
    `;
  }

  div.innerHTML = `
    <span class="compare-rank">${rank}</span>
    <div class="compare-thumb"><img src="${escapeHtml(thumb)}" alt="" loading="lazy" /></div>
    <div class="compare-info">
      <div class="compare-title">${escapeHtml(video.title || '')}</div>
      <div class="compare-channel">${escapeHtml(video.channel_name || video.channel || '')}</div>
      <div class="compare-tags">${tagsHtml}</div>
    </div>
  `;
  return div;
}

// ============================================================
// Watch Page (with auto-tracking)
// ============================================================
async function openWatch(video, pushStateFlag = true) {
  // Stop any currently playing video first
  destroyYTPlayer();

  currentVideo = video;
  navigateTo('watch', pushStateFlag);
  
  if (pushStateFlag) {
    history.pushState({ page: 'watch', videoId: video.video_id }, '', `/watch?v=${video.video_id}`);
  }

  const scores = video.agent_scores || {};
  const vid = video.video_id || '';

  // Resume playback logic
  let startSeconds = 0;
  if (currentUser) {
    try {
      // Check recent history to see if user left off
      const res = await fetch(`${API}/users/${currentUser.id}/history?limit=100`);
      if (res.ok) {
        const hist = await res.json();
        const entry = hist.find(h => h.video_id === vid);
        if (entry && entry.watch_pct && entry.watch_pct < 0.95) { // don't resume if almost finished
          startSeconds = Math.floor(entry.watch_pct * entry.duration_seconds);
        }
      }
    } catch (e) {
      console.error('Failed to fetch history for resume:', e);
    }
  }

  // Use YouTube IFrame API for tracking
  createYTPlayer(vid, startSeconds);

  // Info
  document.getElementById('watch-title').textContent = video.title || '';
  document.getElementById('watch-channel').textContent = video.channel_name || video.channel || '';
  document.getElementById('watch-views').textContent = `${formatCount(video.view_count || 0)} views`;

  // Shield bar
  const bar = document.getElementById('watch-shield-bar');
  bar.innerHTML = `
    <div class="shield-metric"><div class="shield-metric-value" style="color:${scoreHex(scores.shield_score)}">${pct(scores.shield_score)}</div><div class="shield-metric-label">Shield</div></div>
    <div class="shield-metric"><div class="shield-metric-value" style="color:${cbHex(scores.clickbait)}">${pct(scores.clickbait)}</div><div class="shield-metric-label">Clickbait</div></div>
    <div class="shield-metric"><div class="shield-metric-value" style="color:${scoreHex(scores.credibility)}">${pct(scores.credibility)}</div><div class="shield-metric-label">Credibility</div></div>
    <div class="shield-metric"><div class="shield-metric-value" style="color:${scoreHex(scores.info_density)}">${pct(scores.info_density)}</div><div class="shield-metric-label">Info Density</div></div>
    <div class="shield-metric"><div class="shield-metric-value" style="color:${cbHex(scores.emotional_manipulation)}">${pct(scores.emotional_manipulation)}</div><div class="shield-metric-label">Emotion</div></div>
  `;

  // Description
  const desc = video.description || '';
  const descEl = document.getElementById('watch-description');
  descEl.classList.remove('expanded');
  
  if (desc.length > 200) {
    descEl.innerHTML = escapeHtml(desc.substring(0, 200)) + '... <strong>Show more</strong>';
    descEl.style.cursor = 'pointer';
    descEl.onclick = () => {
      const isExpanded = descEl.classList.toggle('expanded');
      if (isExpanded) {
        descEl.innerHTML = escapeHtml(desc) + '<br><br><strong>Show less</strong>';
      } else {
        descEl.innerHTML = escapeHtml(desc.substring(0, 200)) + '... <strong>Show more</strong>';
      }
    };
  } else {
    descEl.textContent = desc;
    descEl.style.cursor = 'default';
    descEl.onclick = null;
  }

  // Hide previous explanation
  document.getElementById('watch-explanation').classList.add('hidden');

  // Explain button
  document.getElementById('watch-explain-btn').onclick = () => explainVideo(video);

  // Update mark button to show auto-save status
  const markBtn = document.getElementById('watch-mark-btn');
  markBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
    ${currentUser ? 'Auto-tracking watch time' : 'Sign in to track history'}
  `;

  // Load recommendations
  loadWatchRecs(video);
}

async function explainVideo(video) {
  const el = document.getElementById('watch-explanation');
  el.classList.remove('hidden');
  el.textContent = 'Generating explanation...';

  try {
    const res = await fetch(`${API}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: video.title || '',
        channel: video.channel_name || video.channel || '',
        description: (video.description || '').substring(0, 300),
        agent_scores: video.agent_scores || {},
        query: document.getElementById('global-search').value || '',
      }),
    });
    const data = await res.json();
    el.textContent = data.explanation || 'No explanation available.';
  } catch {
    el.textContent = 'Failed to generate explanation.';
  }
}

function loadWatchRecs(currentVid) {
  const recsEl = document.getElementById('watch-recs');
  recsEl.innerHTML = '';

  const topic = currentVid.topic || '';
  let recs = feedCache.filter(v =>
    v.video_id !== currentVid.video_id &&
    (v.topic === topic || !topic)
  ).slice(0, 15);

  if (recs.length === 0) recs = feedCache.slice(0, 15);

  recs.forEach(v => {
    const item = document.createElement('div');
    item.className = 'rec-item';
    item.addEventListener('click', () => openWatch(v));

    const scores = v.agent_scores || {};
    const thumb = v.thumbnail_url || v.thumbnail || `https://img.youtube.com/vi/${v.video_id}/mqdefault.jpg`;

    item.innerHTML = `
      <div class="rec-thumb"><img src="${escapeHtml(thumb)}" alt="" loading="lazy" /></div>
      <div class="rec-info">
        <div class="rec-title">${escapeHtml(v.title || '')}</div>
        <div class="rec-channel">${escapeHtml(v.channel_name || v.channel || '')}</div>
        <div class="rec-shield">Shield ${pct(scores.shield_score)}</div>
      </div>
    `;
    recsEl.appendChild(item);
  });
}

// ============================================================
// Trending
// ============================================================
async function loadTrending() {
  const grid = document.getElementById('trending-grid');
  grid.innerHTML = '';

  const sorted = [...feedCache].sort((a, b) =>
    (b.agent_scores?.shield_score || 0) - (a.agent_scores?.shield_score || 0)
  ).slice(0, 30);

  sorted.forEach(v => grid.appendChild(createVideoCard(v)));
}

// ============================================================
// History
// ============================================================
async function loadHistory() {
  const grid = document.getElementById('history-grid');
  const empty = document.getElementById('history-empty');

  if (!currentUser) {
    grid.innerHTML = '';
    empty.classList.remove('hidden');
    empty.querySelector('p').textContent = 'Sign in to see your watch history.';
    return;
  }

  try {
    const res = await fetch(`${API}/users/${currentUser.id}/history?limit=30`);
    if (!res.ok) {
        if (res.status === 404) {
            clearSession();
            grid.innerHTML = '';
            empty.classList.remove('hidden');
            empty.querySelector('p').textContent = 'Session expired. Please sign in again.';
            return;
        }
        throw new Error('Failed to load history');
    }
    const history = await res.json();

    if (!history || history.length === 0) {
      grid.innerHTML = '';
      empty.classList.remove('hidden');
      return;
    }

    empty.classList.add('hidden');
    grid.innerHTML = '';
    history.forEach(h => {
      const card = createVideoCard({
        video_id: h.video_id,
        title: h.title,
        channel_name: h.channel,
        thumbnail: h.thumbnail,
        duration_seconds: h.duration_seconds,
        agent_scores: h.agent_scores,
      });
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = '<div class="empty-state"><p>Failed to load history.</p></div>';
  }
}

// ============================================================
// Settings
// ============================================================
function setupSettings() {
  document.getElementById('save-settings-btn').addEventListener('click', saveSettings);
}

function loadSettings() {
  const card = document.getElementById('settings-card');
  const guest = document.getElementById('settings-guest');

  if (!currentUser) {
    card.classList.add('hidden');
    guest.classList.remove('hidden');
    return;
  }

  card.classList.remove('hidden');
  guest.classList.add('hidden');

  document.getElementById('settings-name').value = currentUser.name || '';
  document.getElementById('settings-email').value = currentUser.email || '';
  document.getElementById('settings-location').value = currentUser.location || '';
  document.getElementById('settings-language').value = currentUser.language || 'en';
  document.getElementById('settings-strictness').value = currentUser.content_strictness || 'balanced';

  const topicsEl = document.getElementById('settings-topics');
  const allTopics = [
    ['machine_learning_ai', 'AI / ML'], ['climate_change', 'Climate'],
    ['investing_finance', 'Finance'], ['history_ancient', 'History'],
    ['healthy_cooking', 'Cooking'], ['fitness_workout', 'Fitness'],
    ['psychology_mental_health', 'Psychology'], ['space_astronomy', 'Space'],
    ['programming_webdev', 'Coding'], ['physics_math', 'Physics'],
    ['music_theory', 'Music'], ['photography_film', 'Film'],
    ['entrepreneurship', 'Startups'], ['language_learning', 'Languages'],
    ['philosophy_ethics', 'Philosophy'], ['biology_medicine', 'Biology'],
    ['gaming_design', 'Gaming'], ['politics_geopolitics', 'Politics'],
    ['art_design', 'Art'], ['diy_engineering', 'DIY'],
  ];

  const userTopics = currentUser.preferred_topics || [];
  topicsEl.innerHTML = allTopics.map(([key, label]) => `
    <label class="topic-chip">
      <input type="checkbox" value="${key}" ${userTopics.includes(key) ? 'checked' : ''} />
      <span>${label}</span>
    </label>
  `).join('');
}

async function saveSettings() {
  if (!currentUser) return;

  const topics = [];
  document.querySelectorAll('#settings-topics input:checked').forEach(cb => topics.push(cb.value));

  try {
    const res = await fetch(`${API}/users/${currentUser.id}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: document.getElementById('settings-name').value.trim(),
        location: document.getElementById('settings-location').value,
        language: document.getElementById('settings-language').value,
        content_strictness: document.getElementById('settings-strictness').value,
        preferred_topics: topics,
      }),
    });
    const data = await res.json();
    if (res.ok) {
      saveSession(data);
      showToast('Settings saved!');
      loadFeed();
    }
  } catch (err) {
    showToast('Failed to save settings.');
  }
}

// ============================================================
// Video Card Component
// ============================================================
function createVideoCard(video) {
  const card = document.createElement('div');
  card.className = 'video-card';
  card.addEventListener('click', () => openWatch(video));

  const scores = video.agent_scores || {};
  const shield = scores.shield_score || 0;
  const thumb = video.thumbnail_url || video.thumbnail || `https://img.youtube.com/vi/${video.video_id}/mqdefault.jpg`;
  const dur = formatDuration(video.duration_seconds || 0);
  const channel = video.channel_name || video.channel || '';
  const views = formatCount(video.view_count || 0);
  const initial = channel ? channel[0].toUpperCase() : '?';

  const badgeClass = shield >= 0.65 ? 'green' : shield >= 0.4 ? 'amber' : 'red';

  card.innerHTML = `
    <div class="card-thumbnail">
      <img src="${escapeHtml(thumb)}" alt="${escapeHtml(video.title || '')}" loading="lazy" />
      <span class="card-duration">${dur}</span>
      <span class="card-shield-badge ${badgeClass}">${Math.round(shield * 100)}</span>
    </div>
    <div class="card-meta">
      <div class="card-avatar">${initial}</div>
      <div class="card-info">
        <div class="card-title">${escapeHtml(video.title || '')}</div>
        <div class="card-channel">${escapeHtml(channel)}</div>
        <div class="card-stats">${views} views &middot; ${video.topic || ''}</div>
      </div>
    </div>
  `;

  return card;
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
  const el = document.createElement('span');
  el.textContent = str;
  return el.innerHTML;
}

function pct(val) {
  return Math.round((val || 0) * 100) + '%';
}

function formatDuration(seconds) {
  if (!seconds) return '0:00';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatCount(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

function scoreColor(val) {
  val = val || 0;
  if (val >= 0.65) return 'green';
  if (val >= 0.4) return 'amber';
  return 'red';
}

function cbColor(val) {
  val = val || 0;
  if (val <= 0.3) return 'green';
  if (val <= 0.6) return 'amber';
  return 'red';
}

function scoreHex(val) {
  val = val || 0;
  if (val >= 0.65) return '#3ccf4e';
  if (val >= 0.4) return '#f5c518';
  return '#ff4e45';
}

function cbHex(val) {
  val = val || 0;
  if (val <= 0.3) return '#3ccf4e';
  if (val <= 0.6) return '#f5c518';
  return '#ff4e45';
}
