const state = {
  activeTab: "overview",
  snapshot: null,
  snapshotMeta: {
    status: "idle",
    fetched_at: null,
    cache_hit: false,
    refreshing: false,
    last_error: null,
  },
  jobs: [],
  settings: null,
  currentDrilldown: null,
  jobsPollHandle: null,
  globalJobsPollHandle: null,
  snapshotPollHandle: null,
  reloadDismissedAt: null,
  concentrationSorts: {
    sector: { key: "weight", direction: "desc" },
    exchange: { key: "weight", direction: "desc" },
  },
};

const elements = {
  tabs: () => Array.from(document.querySelectorAll(".tab")),
  tabContent: () => document.getElementById("tab-content"),
  loading: () => document.getElementById("loading"),
  errorBanner: () => document.getElementById("error-banner"),
  macroAlert: () => document.getElementById("macro-alert"),
  modeAlert: () => document.getElementById("mode-alert"),
  reloadAlert: () => document.getElementById("reload-alert"),
  status: () => document.getElementById("snapshot-status"),
  context: () => document.getElementById("snapshot-context"),
  drilldown: () => document.getElementById("drilldown-panel"),
  reportViewer: () => document.getElementById("report-viewer"),
  reportViewerTitle: () => document.getElementById("report-viewer-title"),
  reportViewerBody: () => document.getElementById("report-viewer-body"),
  reportViewerClose: () => document.getElementById("report-viewer-close"),
  refreshButton: () => document.getElementById("refresh-portfolio-btn"),
};

function setLoading(isLoading) {
  elements.loading().classList.toggle("hidden", !isLoading);
}

function setError(message) {
  const banner = elements.errorBanner();
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.classList.remove("hidden");
  banner.textContent = message;
}

function loadingSnapshotMessage() {
  if (state.settings?.read_only === false) {
    return "Fetching IBKR positions, watchlist, and orders; may take a few minutes.";
  }
  return "Loading current data…";
}
