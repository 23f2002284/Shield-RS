/**
 * Shield — Frontend Application
 * Connects to FastAPI backend at /api/*
 */

const API = ""; // Uses Vite proxy in dev - forwards /api/* to localhost:8000

// -- State --
let currentUser = JSON.parse(localStorage.getItem("shield_user") || "null");
let currentPage = "search";
let lastSearchQuery = ""; // Syncs search query to compare tab

// -- DOM Refs --
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// -- Helpers --
function formatDuration(seconds) {
  if (!seconds) return "";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function formatCount(n) {
  if (!n || n === 0) return "0";
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return String(n);
}

function scoreColor(score) {
  if (score >= 0.6) return "green";
  if (score >= 0.35) return "amber";
  return "red";
}

function scoreLabel(score) {
  return Math.round(score * 100);
}

function getThumbnail(video) {
  if (video.thumbnail) return video.thumbnail;
  if (video.thumbnail_url) return video.thumbnail_url;
  const vid = video.video_id || "";
  return `https://img.youtube.com/vi/${vid}/mqdefault.jpg`;
}

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error(`API error: ${path}`, err);
    return null;
  }
}

// -- Navigation --
function navigateTo(page) {
  currentPage = page;
  $$(".page").forEach((p) => p.classList.remove("active"));
  $(`#page-${page}`).classList.add("active");
  $$(".nav-link").forEach((n) => n.classList.remove("active"));
  $(`#nav-${page}`).classList.add("active");

  if (page === "profile") renderProfile();

  // Sync search query to compare tab
  if (page === "compare" && lastSearchQuery) {
    const compareInput = $("#compare-input");
    if (compareInput && !compareInput.value.trim()) {
      compareInput.value = lastSearchQuery;
      // Auto-trigger compare if we have a query
      performCompare();
    }
  }
}

// -- Video Card --
function createVideoCard(video) {
  const scores = video.agent_scores || {};
  const shield = scores.shield_score || 0;
  const card = document.createElement("div");
  card.className = "video-card";
  card.innerHTML = `
    <div class="card-thumbnail">
      <img src="${getThumbnail(video)}" alt="" loading="lazy" 
           onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 320 180%22%3E%3Crect width=%22320%22 height=%22180%22 fill=%22%23e0e0e0%22/%3E%3Ctext x=%22160%22 y=%2296%22 text-anchor=%22middle%22 fill=%22%23999%22 font-size=%2214%22%3ENo thumbnail%3C/text%3E%3C/svg%3E'" />
      ${video.duration_seconds ? `<span class="card-duration">${formatDuration(video.duration_seconds)}</span>` : ""}
      <div class="card-shield-badge ${scoreColor(shield)}">${scoreLabel(shield)}</div>
    </div>
    <div class="card-body">
      <div class="card-title">${escapeHtml(video.title || "Untitled")}</div>
      <div class="card-channel">${escapeHtml(video.channel_title || video.channel_name || video.channel || "")}</div>
      <div class="card-scores">
        ${makeTag("Credibility", scores.credibility)}
        ${makeTag("Clickbait", scores.clickbait, true)}
        ${makeTag("Info", scores.info_density)}
      </div>
    </div>
  `;
  card.addEventListener("click", () => openVideoModal(video));
  return card;
}

function makeTag(label, score, inverted = false) {
  if (score === undefined) return "";
  const val = Math.round(score * 100);
  let color;
  if (inverted) {
    color = score <= 0.3 ? "green" : score <= 0.6 ? "amber" : "red";
  } else {
    color = scoreColor(score);
  }
  return `<span class="score-tag ${color}">${label} ${val}%</span>`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// -- Video List Item for YOUTUBE column (shows view count, likes, subs) --
function createYouTubeListItem(video, rank) {
  const item = document.createElement("div");
  item.className = "video-list-item";
  const views = video.view_count || 0;
  const likes = video.like_count || 0;
  const subs = video.subscriber_count || 0;
  const channel = video.channel_title || video.channel_name || video.channel || "";
  item.innerHTML = `
    <div class="list-thumbnail">
      <img src="${getThumbnail(video)}" alt="" loading="lazy"
           onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 160 90%22%3E%3Crect width=%22160%22 height=%2290%22 fill=%22%23e0e0e0%22/%3E%3C/svg%3E'" />
    </div>
    <div class="list-info">
      <div class="list-title">${rank ? `#${rank} ` : ""}${escapeHtml(video.title || "Untitled")}</div>
      <div class="list-channel">${escapeHtml(channel)}${video.duration_seconds ? ` · ${formatDuration(video.duration_seconds)}` : ""}</div>
      <div class="list-scores">
        <span class="score-tag yt-views">${formatCount(views)} views</span>
        <span class="score-tag yt-likes">${formatCount(likes)} likes</span>
        <span class="score-tag yt-subs">${formatCount(subs)} subs</span>
      </div>
    </div>
  `;
  item.addEventListener("click", () => openVideoModal(video));
  return item;
}

// -- Video List Item for SHIELD column (shows agent scores) --
function createShieldListItem(video, rank) {
  const scores = video.agent_scores || {};
  const shield = scores.shield_score || 0;
  const channel = video.channel_title || video.channel_name || video.channel || "";
  const item = document.createElement("div");
  item.className = "video-list-item";
  item.innerHTML = `
    <div class="list-thumbnail">
      <img src="${getThumbnail(video)}" alt="" loading="lazy"
           onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 160 90%22%3E%3Crect width=%22160%22 height=%2290%22 fill=%22%23e0e0e0%22/%3E%3C/svg%3E'" />
    </div>
    <div class="list-info">
      <div class="list-title">${rank ? `#${rank} ` : ""}${escapeHtml(video.title || "Untitled")}</div>
      <div class="list-channel">${escapeHtml(channel)}${video.duration_seconds ? ` · ${formatDuration(video.duration_seconds)}` : ""}</div>
      <div class="list-scores">
        <span class="score-tag ${scoreColor(shield)}">Shield ${scoreLabel(shield)}</span>
        ${makeTag("Cred", scores.credibility)}
        ${makeTag("CB", scores.clickbait, true)}
        ${makeTag("Info", scores.info_density)}
      </div>
    </div>
  `;
  item.addEventListener("click", () => openVideoModal(video));
  return item;
}

// -- Generic list item (for history) --
function createVideoListItem(video, rank) {
  return createShieldListItem(video, rank);
}

// -- Video Modal --
let _currentModalVideo = null;

function openVideoModal(video) {
  _currentModalVideo = video;
  const modal = $("#video-modal");
  const scores = video.agent_scores || {};

  // Player
  const videoId = video.video_id || "";
  $("#modal-player").innerHTML = `
    <iframe src="https://www.youtube.com/embed/${videoId}?autoplay=1" 
            allow="autoplay; encrypted-media" allowfullscreen></iframe>
  `;

  $("#modal-title").textContent = video.title || "Untitled";
  $("#modal-channel").textContent = video.channel_title || video.channel_name || video.channel || "";
  $("#modal-description").textContent = video.description || "No description available.";

  // Scores
  const scoresEl = $("#modal-scores");
  scoresEl.innerHTML = "";
  const scoreItems = [
    { key: "shield_score", label: "Shield Score", value: scores.shield_score },
    { key: "credibility", label: "Credibility", value: scores.credibility },
    { key: "info_density", label: "Info Density", value: scores.info_density },
    { key: "clickbait", label: "Clickbait", value: scores.clickbait, inverted: true },
    { key: "emotional_manipulation", label: "Emotion", value: scores.emotional_manipulation, inverted: true },
    { key: "goal_alignment", label: "Relevance", value: scores.goal_alignment },
  ];

  scoreItems.forEach((s) => {
    if (s.value === undefined) return;
    const val = Math.round(s.value * 100);
    const color = s.inverted ? (s.value <= 0.3 ? "green" : s.value <= 0.6 ? "amber" : "red") : scoreColor(s.value);
    const el = document.createElement("div");
    el.className = "modal-score-item";
    el.innerHTML = `
      <div class="modal-score-value" style="color: var(--${color})">${val}</div>
      <div class="modal-score-label">${s.label}</div>
    `;
    scoresEl.appendChild(el);
  });

  // Reset explanation
  const explainEl = $("#modal-explanation");
  explainEl.textContent = "";
  explainEl.classList.add("hidden");
  const explainBtn = $("#explain-btn");
  explainBtn.textContent = "Why this score?";
  explainBtn.disabled = false;
  explainBtn.classList.remove("hidden");

  // Watch button
  const watchBtn = $("#modal-watch-btn");
  watchBtn.onclick = () => logWatch(video);

  modal.classList.remove("hidden");
}

async function explainCurrentVideo() {
  if (!_currentModalVideo) return;
  const btn = $("#explain-btn");
  const explainEl = $("#modal-explanation");

  btn.textContent = "Thinking...";
  btn.disabled = true;
  explainEl.classList.remove("hidden");
  explainEl.textContent = "Generating explanation...";

  const video = _currentModalVideo;
  const data = await apiFetch("/api/explain", {
    method: "POST",
    body: JSON.stringify({
      title: video.title || "",
      channel: video.channel_title || video.channel_name || video.channel || "",
      description: (video.description || "").slice(0, 500),
      agent_scores: video.agent_scores || {},
      query: lastSearchQuery,
    }),
  });

  if (data && data.explanation) {
    explainEl.textContent = data.explanation;
    btn.textContent = "Explanation generated";
  } else {
    explainEl.textContent = "Could not generate explanation. Please try again.";
    btn.textContent = "Why this score?";
    btn.disabled = false;
  }
}

function closeModal() {
  const modal = $("#video-modal");
  modal.classList.add("hidden");
  $("#modal-player").innerHTML = "";
  _currentModalVideo = null;
}

// -- Search --
async function performSearch() {
  const query = $("#search-input").value.trim();
  if (!query) return;

  lastSearchQuery = query; // Sync to compare tab

  const timeBudget = parseInt($("#time-budget").value);
  const qualityPref = $("#quality-pref").value;
  const maxResults = parseInt($("#max-results").value);

  // Show loading
  $("#search-results").innerHTML = "";
  $("#search-loading").classList.remove("hidden");
  $("#search-metrics").classList.add("hidden");
  $("#search-empty").classList.add("hidden");

  const data = await apiFetch("/api/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      max_results: maxResults,
      time_budget_minutes: timeBudget,
      quality_preference: qualityPref,
    }),
  });

  $("#search-loading").classList.add("hidden");

  if (!data || !data.videos || data.videos.length === 0) {
    $("#search-empty").classList.remove("hidden");
    return;
  }

  // Metrics
  const m = data.metrics || {};
  $("#metric-results").textContent = m.total_results || 0;
  $("#metric-shield").textContent = m.avg_shield_score || 0;
  $("#metric-clickbait").textContent = m.avg_clickbait || 0;
  $("#metric-credibility").textContent = m.avg_credibility || 0;
  $("#metric-time").textContent = `${data.elapsed_ms || 0}ms`;
  $("#search-metrics").classList.remove("hidden");

  // Render cards
  const grid = $("#search-results");
  data.videos.forEach((v) => grid.appendChild(createVideoCard(v)));
}

// -- Compare --
async function performCompare() {
  const query = $("#compare-input").value.trim();
  if (!query) return;

  lastSearchQuery = query; // Keep in sync

  $("#compare-loading").classList.remove("hidden");
  $("#compare-container").classList.add("hidden");

  const data = await apiFetch("/api/compare", {
    method: "POST",
    body: JSON.stringify({ query, max_results: 15 }),
  });

  $("#compare-loading").classList.add("hidden");
  if (!data) return;

  $("#compare-container").classList.remove("hidden");

  // YouTube side - show view count, likes, subs
  const ytMetrics = data.youtube?.metrics || {};
  $("#youtube-metrics").innerHTML = `
    <div class="compare-metric">
      <div class="compare-metric-value">${ytMetrics.avg_clickbait || 0}</div>
      <div class="compare-metric-label">Avg Clickbait</div>
    </div>
    <div class="compare-metric">
      <div class="compare-metric-value">${ytMetrics.avg_credibility || 0}</div>
      <div class="compare-metric-label">Avg Credibility</div>
    </div>
    <div class="compare-metric">
      <div class="compare-metric-value">${ytMetrics.avg_shield_score || 0}</div>
      <div class="compare-metric-label">Avg Shield</div>
    </div>
  `;
  const ytList = $("#youtube-results");
  ytList.innerHTML = "";
  (data.youtube?.videos || []).forEach((v, i) =>
    ytList.appendChild(createYouTubeListItem(v, i + 1))
  );

  // Shield side - show agent scores
  const shMetrics = data.shield?.metrics || {};
  $("#shield-metrics").innerHTML = `
    <div class="compare-metric">
      <div class="compare-metric-value" style="color: var(--green)">${shMetrics.avg_shield_score || 0}</div>
      <div class="compare-metric-label">Avg Shield</div>
    </div>
    <div class="compare-metric">
      <div class="compare-metric-value">${shMetrics.avg_clickbait || 0}</div>
      <div class="compare-metric-label">Avg Clickbait</div>
    </div>
    <div class="compare-metric">
      <div class="compare-metric-value">${shMetrics.avg_credibility || 0}</div>
      <div class="compare-metric-label">Avg Credibility</div>
    </div>
  `;
  const shList = $("#shield-results");
  shList.innerHTML = "";
  (data.shield?.videos || []).forEach((v, i) =>
    shList.appendChild(createShieldListItem(v, i + 1))
  );
}

// -- Profile --
function renderProfile() {
  if (currentUser) {
    $("#profile-avatar").textContent = currentUser.name[0].toUpperCase();
    $("#profile-name").textContent = currentUser.name;
    $("#profile-joined").textContent = `Joined ${new Date(currentUser.created_at).toLocaleDateString()}`;
    $("#profile-setup").classList.add("hidden");
    $("#profile-prefs").classList.remove("hidden");
    $("#history-section").classList.remove("hidden");
    $("#recommend-section").classList.remove("hidden");
    loadHistory();
    loadRecommendations();
  } else {
    $("#profile-avatar").textContent = "?";
    $("#profile-name").textContent = "Guest";
    $("#profile-joined").textContent = "Not signed in";
    $("#profile-setup").classList.remove("hidden");
    $("#profile-prefs").classList.add("hidden");
    $("#history-section").classList.add("hidden");
    $("#recommend-section").classList.add("hidden");
  }
  updateUserBadge();
}

async function createUserProfile() {
  const name = $("#new-user-name").value.trim();
  if (!name) return;

  const user = await apiFetch("/api/users", {
    method: "POST",
    body: JSON.stringify({ name }),
  });

  if (user) {
    currentUser = user;
    localStorage.setItem("shield_user", JSON.stringify(user));
    renderProfile();
  }
}

async function loadHistory() {
  if (!currentUser) return;

  const history = await apiFetch(`/api/users/${currentUser.id}/history?limit=20`);
  const list = $("#history-list");
  const empty = $("#history-empty");
  list.innerHTML = "";

  if (!history || history.length === 0) {
    empty.classList.remove("hidden");
    return;
  }

  empty.classList.add("hidden");
  history.forEach((h) => {
    const item = createVideoListItem({
      video_id: h.video_id,
      title: h.title,
      channel: h.channel,
      thumbnail: h.thumbnail,
      duration_seconds: h.duration_seconds,
      agent_scores: h.agent_scores,
    });
    list.appendChild(item);
  });
}

async function loadRecommendations() {
  if (!currentUser) return;

  const data = await apiFetch(`/api/users/${currentUser.id}/recommend?top_k=8`);
  const grid = $("#recommend-grid");
  grid.innerHTML = "";

  if (!data || !data.recommendations || data.recommendations.length === 0) {
    grid.innerHTML = '<div class="empty-state"><p>Watch some videos first to get personalized recommendations.</p></div>';
    return;
  }

  data.recommendations.forEach((v) => grid.appendChild(createVideoCard(v)));
}

async function logWatch(video) {
  if (!currentUser) {
    alert("Create a profile first to track your watch history!");
    return;
  }

  const scores = video.agent_scores || {};
  await apiFetch(`/api/users/${currentUser.id}/history`, {
    method: "POST",
    body: JSON.stringify({
      video_id: video.video_id || "",
      title: video.title || "",
      channel: video.channel_title || video.channel_name || video.channel || "",
      thumbnail: getThumbnail(video),
      duration_seconds: video.duration_seconds || 0,
      watch_pct: 1.0,
      agent_scores: scores,
    }),
  });

  // Visual feedback
  const btn = $("#modal-watch-btn");
  btn.textContent = "Marked as watched!";
  btn.style.background = "var(--green)";
  setTimeout(() => {
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      Mark as Watched
    `;
    btn.style.background = "";
  }, 2000);
}

function updateUserBadge() {
  if (currentUser) {
    $("#user-avatar").textContent = currentUser.name[0].toUpperCase();
    $("#user-name-display").textContent = currentUser.name;
  } else {
    $("#user-avatar").textContent = "?";
    $("#user-name-display").textContent = "Guest";
  }
}

async function savePreferences() {
  if (!currentUser) return;
  const style = $("#pref-style").value;
  await apiFetch(`/api/users/${currentUser.id}/prefs`, {
    method: "PUT",
    body: JSON.stringify({ preferences: { style } }),
  });
  currentUser.preferences = { style };
  localStorage.setItem("shield_user", JSON.stringify(currentUser));
}

// -- Event Listeners --
document.addEventListener("DOMContentLoaded", () => {
  // Navigation
  $$(".nav-link").forEach((link) => {
    link.addEventListener("click", () => navigateTo(link.dataset.page));
  });

  $("#logo-link").addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("search");
  });

  // User badge -> profile
  $("#user-badge").addEventListener("click", () => navigateTo("profile"));

  // Search
  $("#search-btn").addEventListener("click", performSearch);
  $("#search-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") performSearch();
  });

  // Compare
  $("#compare-btn").addEventListener("click", performCompare);
  $("#compare-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") performCompare();
  });

  // Profile
  $("#create-user-btn").addEventListener("click", createUserProfile);
  $("#save-prefs-btn").addEventListener("click", savePreferences);

  // Modal
  $("#modal-close").addEventListener("click", closeModal);
  $("#video-modal").addEventListener("click", (e) => {
    if (e.target === $("#video-modal")) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Explain button
  $("#explain-btn").addEventListener("click", explainCurrentVideo);

  // Init
  updateUserBadge();
});
