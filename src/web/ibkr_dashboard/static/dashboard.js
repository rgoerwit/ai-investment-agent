
function renderActiveTab() {
  if (!state.snapshot && state.activeTab !== "settings") {
    const message =
      state.snapshotMeta.status === "loading"
        ? loadingSnapshotMessage()
        : state.snapshotMeta.status === "error"
          ? "Current data unavailable. Use Reload Data to retry."
          : "No current data loaded yet. It should load automatically in a moment.";
    elements.tabContent().innerHTML = `<p class="muted">${escapeHtml(message)}</p>`;
    return;
  }
  const content = {
    overview: renderOverview,
    actions: renderActions,
    watchlist: renderWatchlist,
    orders: renderOrders,
    refresh: renderRefresh,
    settings: renderSettings,
  }[state.activeTab]();
  elements.tabContent().innerHTML = content;
  bindDynamicHandlers();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    const error = new Error(payload.message || payload.error || "Request failed");
    error.payload = payload;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function stopSnapshotPolling() {
  if (state.snapshotPollHandle) {
    clearTimeout(state.snapshotPollHandle);
    state.snapshotPollHandle = null;
  }
}

function scheduleSnapshotPoll() {
  stopSnapshotPolling();
  state.snapshotPollHandle = setTimeout(() => {
    loadPortfolio(false);
  }, 2000);
}

function applySnapshotPayload(payload) {
  state.snapshot = payload.portfolio ? payload : null;
  state.snapshotMeta = {
    status: payload.status || (payload.portfolio ? "ready" : "idle"),
    fetched_at: payload.as_of || payload.fetched_at || null,
    cache_hit: Boolean(payload.cache_hit),
    refreshing: Boolean(payload.refreshing),
    last_error: payload.load_error || payload.last_error || null,
  };
  updateMacroAlert();
  updateStatus();
  renderActiveTab();

  if (state.snapshotMeta.status === "loading" || state.snapshotMeta.refreshing) {
    scheduleSnapshotPoll();
  } else {
    stopSnapshotPolling();
  }
}

async function loadPortfolio(force = false) {
  setLoading(true);
  if (!force) {
    setError(null);
  }
  try {
    const suffix = force ? "?refresh=1" : "";
    const payload = await fetchJson(`/api/portfolio${suffix}`);
    applySnapshotPayload(payload);
    setError(payload.load_error || null);
  } catch (error) {
    stopSnapshotPolling();
    if (error.payload) {
      state.snapshot = null;
      state.snapshotMeta = {
        status: error.payload.status || "error",
        fetched_at: error.payload.fetched_at || null,
        cache_hit: Boolean(error.payload.cache_hit),
        refreshing: Boolean(error.payload.refreshing),
        last_error: error.payload.last_error || error.payload.message || error.message,
      };
      updateMacroAlert();
      updateStatus();
      renderActiveTab();
      setError(state.snapshotMeta.last_error);
    } else {
      setError(error.message);
    }
  } finally {
    setLoading(false);
  }
}

async function loadJobs() {
  try {
    const payload = await fetchJson("/api/refresh/jobs");
    state.jobs = payload.jobs || [];
    if (state.activeTab === "refresh") {
      renderActiveTab();
    }
  } catch (error) {
    setError(error.message);
  }
}

function stopJobsPolling() {
  if (state.jobsPollHandle) {
    clearInterval(state.jobsPollHandle);
    state.jobsPollHandle = null;
  }
}

function syncJobsPolling() {
  stopJobsPolling();
  if (state.activeTab !== "refresh") {
    return;
  }
  loadJobs();
  state.jobsPollHandle = setInterval(loadJobs, 5000);
}

async function loadSettings() {
  try {
    state.settings = await fetchJson("/api/settings");
    updateModeAlert();
    if (state.activeTab === "settings") {
      renderActiveTab();
    }
  } catch (error) {
    setError(error.message);
  }
}

async function loadDrilldown(ticker) {
  elements.drilldown().innerHTML = "<p class='muted'>Loading drilldown…</p>";
  try {
    const payload = await fetchJson(`/api/equities/${encodeURIComponent(ticker)}`);
    if (payload.status === "loading") {
      elements.drilldown().innerHTML =
        "<p class='muted'>Current data is still loading. Try again in a moment.</p>";
      return;
    }
    state.currentDrilldown = payload;
    elements.drilldown().innerHTML = renderDrilldown(payload);
    bindReportHandlers();
  } catch (error) {
    setError(error.message);
  }
}

function updateMacroAlert() {
  const alert = elements.macroAlert();
  const macro = state.snapshot?.macro_alert;
  if (!macro?.detected) {
    alert.classList.add("hidden");
    alert.textContent = "";
    return;
  }
  alert.classList.remove("hidden");
  const escapedHeadline = macro.headline ? escapeHtml(macro.headline) : null;
  const headline = escapedHeadline
    ? `Headline: ${escapedHeadline}`
    : "Macro event detected.";
  alert.innerHTML = `<strong>Macro alert:</strong> ${headline} (${escapeHtml(String(macro.correlation_pct || "—"))}% of held positions)`;
}

function updateModeAlert() {
  const alert = elements.modeAlert();
  const snapshot = state.snapshot;
  const settings = state.settings || {};
  if (!snapshot?.read_only) {
    alert.classList.add("hidden");
    alert.textContent = "";
    return;
  }
  const accountHint = settings.account_id
    ? ` Current account override: <code>${escapeHtml(settings.account_id)}</code>.`
    : "";
  const resultsDir = escapeHtml(String(settings.results_dir || "results/"));
  alert.classList.remove("hidden");
  alert.innerHTML =
    `<strong>Read-only data view:</strong> this dashboard is showing saved analysis results from <code>${resultsDir}</code>, not your live IBKR portfolio. Switch Data source to live in Settings, or restart with <code>--live</code> / <code>IBKR_DASHBOARD_READ_ONLY=false</code> and working broker credentials.`
    + accountHint;
}

function updateStatus() {
  const status = elements.status();
  const context = elements.context();
  if (state.snapshotMeta.status === "loading") {
    status.textContent =
      state.settings?.read_only === false ? "Loading live data…" : "Loading data…";
    context.textContent = loadingSnapshotMessage();
    return;
  }
  if (state.snapshotMeta.status === "error") {
    status.textContent = "Data load failed";
    context.textContent = "";
    return;
  }
  if (!state.snapshot) {
    status.textContent = "No data loaded";
    const settings = state.settings || {};
    const parts = [];
    if (settings.account_id) parts.push(`Account ${settings.account_id}`);
    if (settings.watchlist_name) parts.push(`Watchlist ${settings.watchlist_name}`);
    parts.push(settings.read_only ? "Read-only mode" : "Live IBKR mode");
    context.textContent = parts.join(" • ");
    updateModeAlert();
    return;
  }
  const freshness = state.snapshotMeta.refreshing ? "refreshing" : "ready";
  const source = state.snapshot.cache_hit ? "cached" : "loaded";
  const mode = state.snapshot.read_only ? "read-only" : "live";
  status.textContent = `Data ${source} at ${state.snapshot.as_of} (${freshness}, ${mode})`;
  const parts = [];
  if (state.snapshot.portfolio?.account_id) {
    parts.push(`Account ${state.snapshot.portfolio.account_id}`);
  }
  if (state.snapshot.watchlist?.name) {
    parts.push(`Watchlist ${state.snapshot.watchlist.name}`);
  }
  parts.push(state.snapshot.read_only ? "Read-only results view" : "Live IBKR data");
  context.textContent = parts.join(" • ");
}

async function createJob(scope, tickers = []) {
  if (scope === "ticker_list" && !tickers.length) {
    setError("Enter at least one ticker before queueing a ticker-list refresh job.");
    return;
  }
  try {
    const payload = {
      scope,
      tickers,
      quick_mode: state.settings?.quick_mode_default ?? true,
      refresh_limit: state.settings?.refresh_limit ?? 10,
      max_age_days: state.settings?.max_age_days ?? 14,
    };
    await fetchJson("/api/refresh/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadJobs();
  } catch (error) {
    setError(error.message);
  }
}

function bindReportHandlers() {
  document.querySelectorAll(".report-open[data-report-kind]").forEach((button) => {
    button.addEventListener("click", () => openReportViewer(button.dataset.reportKind));
  });
}

function bindDynamicHandlers() {
  document.querySelectorAll(".ticker-link[data-ticker]").forEach((button) => {
    button.addEventListener("click", () => loadDrilldown(button.dataset.ticker));
  });

  bindReportHandlers();

  document.querySelectorAll(".sort-button[data-sort-section][data-sort-key]").forEach(
    (button) => {
      button.addEventListener("click", () => {
        const { sortSection, sortKey } = button.dataset;
        const current = state.concentrationSorts[sortSection] || {
          key: "weight",
          direction: "desc",
        };
        const nextDirection =
          current.key === sortKey
            ? current.direction === "asc"
              ? "desc"
              : "asc"
            : getDefaultConcentrationDirection(sortKey);
        state.concentrationSorts[sortSection] = {
          key: sortKey,
          direction: nextDirection,
        };
        renderActiveTab();
      });
    },
  );

  const staleButton = document.getElementById("job-stale");
  if (staleButton) {
    staleButton.addEventListener("click", () => createJob("stale_positions"));
  }
  const dueSoonButton = document.getElementById("job-due-soon");
  if (dueSoonButton) {
    dueSoonButton.addEventListener("click", () => createJob("due_soon"));
  }
  const customButton = document.getElementById("job-custom");
  if (customButton) {
    customButton.addEventListener("click", () => {
      const value = document.getElementById("job-ticker-input").value;
      const tickers = value
        .split(",")
        .map((ticker) => ticker.trim())
        .filter(Boolean);
      createJob("ticker_list", tickers);
    });
  }

  const settingsForm = document.getElementById("settings-form");
  if (settingsForm) {
    settingsForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData(settingsForm);
      const payload = {
        account_id: formData.get("account_id") || null,
        watchlist_name: formData.get("watchlist_name") || null,
        read_only: formData.get("read_only") === "true",
        max_age_days: Number(formData.get("max_age_days") || 14),
        refresh_limit: Number(formData.get("refresh_limit") || 10),
        quick_mode_default: formData.get("quick_mode_default") === "true",
        notes: formData.get("notes") || "",
      };
      try {
        state.settings = await fetchJson("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        updateModeAlert();
        renderActiveTab();
        if (state.settings.snapshot_reload_required) {
          await loadPortfolio(false);
        } else {
          updateStatus();
        }
      } catch (error) {
        setError(error.message);
      }
    });
  }
}

function setActiveTab(name) {
  state.activeTab = name;
  elements.tabs().forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  renderActiveTab();
  syncJobsPolling();
}

function bindStaticHandlers() {
  elements.tabs().forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });
  elements.refreshButton().addEventListener("click", () => loadPortfolio(true));
  elements.reportViewerClose().addEventListener("click", closeReportViewer);
  elements.reportViewer().addEventListener("click", (event) => {
    if (event.target?.dataset?.reportClose === "true") {
      closeReportViewer();
    }
  });
}

async function initializeDashboard() {
  bindStaticHandlers();
  await loadSettings();
  await loadPortfolio(false);
  syncJobsPolling();
}

initializeDashboard();
