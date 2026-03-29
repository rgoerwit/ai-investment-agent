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
  jobsPollHandle: null,
  snapshotPollHandle: null,
};

const elements = {
  tabs: () => Array.from(document.querySelectorAll(".tab")),
  tabContent: () => document.getElementById("tab-content"),
  loading: () => document.getElementById("loading"),
  errorBanner: () => document.getElementById("error-banner"),
  macroAlert: () => document.getElementById("macro-alert"),
  modeAlert: () => document.getElementById("mode-alert"),
  status: () => document.getElementById("snapshot-status"),
  context: () => document.getElementById("snapshot-context"),
  drilldown: () => document.getElementById("drilldown-panel"),
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

function fmtCurrency(value) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function fmtLocalMoney(value, currency = "") {
  if (value === null || value === undefined) return "—";
  return `${Number(value).toFixed(2)}${currency ? ` ${currency}` : ""}`;
}

function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
}

function fmtPct(value) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${Number(value).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatDetailValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (Array.isArray(value)) {
    if (!value.length) return "—";
    if (value.every((item) => typeof item !== "object")) {
      return escapeHtml(value.join(", "));
    }
  }
  if (typeof value === "object") {
    return `<pre class="detail-pre">${escapeHtml(
      JSON.stringify(value, null, 2),
    )}</pre>`;
  }
  return escapeHtml(value);
}

function renderCards(cards) {
  return `<div class="cards">${cards
    .map(
      (card) => `
        <article class="card">
          <div class="label">${escapeHtml(card.label)}</div>
          <div class="value">${escapeHtml(card.value)}</div>
        </article>
      `,
    )
    .join("")}</div>`;
}

function renderTickerLink(item) {
  const ticker = item.ticker_yf || item.ticker_ibkr;
  return `<button type="button" class="ticker-link" data-ticker="${escapeHtml(ticker)}">${escapeHtml(item.ticker_ibkr || ticker)}</button>`;
}

function renderActionTable(title, items, extraColumns = []) {
  if (!items || !items.length) {
    return `<section><h3 class="section-title">${escapeHtml(title)}</h3><p class="muted">None.</p></section>`;
  }
  const headers = ["Ticker", "Action", "Reason", ...extraColumns.map((col) => col.label)];
  const rows = items
    .map((item) => {
      const cells = [
        renderTickerLink(item),
        escapeHtml(item.action),
        escapeHtml(item.reason),
        ...extraColumns.map((col) => col.render(item)),
      ];
      return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
    })
    .join("");
  return `
    <section>
      <h3 class="section-title">${escapeHtml(title)}</h3>
      <table>
        <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function renderCandidatePreview(title, items, limit = 5) {
  if (!items || !items.length) {
    return "";
  }
  const previewItems = items.slice(0, limit);
  const rows = previewItems
    .map(
      (item) => `
        <tr>
          <td>${renderTickerLink(item)}</td>
          <td>${escapeHtml(item.reason || "—")}</td>
          <td>${escapeHtml(item.analysis?.health_adj ?? "—")}</td>
          <td>${escapeHtml(item.analysis?.growth_adj ?? "—")}</td>
        </tr>
      `,
    )
    .join("");
  const hiddenCount = items.length - previewItems.length;
  const footer =
    hiddenCount > 0
      ? `<p class="muted">${hiddenCount} more names are available in Watchlist & Candidates.</p>`
      : "<p class='muted'>Open Watchlist & Candidates for drilldowns and the full list.</p>";
  return `
    <section>
      <h3 class="section-title">${escapeHtml(title)}</h3>
      <table>
        <thead><tr><th>Ticker</th><th>Reason</th><th>Health</th><th>Growth</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${footer}
    </section>
  `;
}

function renderCashTimelineTable(rows, emptyMessage) {
  const body = (rows || [])
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.settlement_date || "—")}</td>
          <td>${escapeHtml(row.ticker_ibkr)}</td>
          <td>${fmtCurrency(row.cash_impact_usd)}</td>
        </tr>
      `,
    )
    .join("");
  return `
    <table>
      <thead><tr><th>Settlement</th><th>Ticker</th><th>USD</th></tr></thead>
      <tbody>${body || `<tr><td colspan="3" class="muted">${escapeHtml(emptyMessage)}</td></tr>`}</tbody>
    </table>
  `;
}

function renderInlineMetrics(title, rows) {
  return `
    <article class="summary-panel">
      <h3>${escapeHtml(title)}</h3>
      <div class="inline-metrics">
        ${rows
          .map(
            (row) => `
              <div class="inline-metric">
                <span class="label">${escapeHtml(row.label)}</span>
                <span class="value">${escapeHtml(String(row.value))}</span>
              </div>
            `,
          )
          .join("")}
      </div>
    </article>
  `;
}

function renderOverview() {
  const snapshot = state.snapshot;
  const portfolio = snapshot.portfolio;
  const overview = snapshot.overview || {};
  const cashSummary = snapshot.cash_summary || {};
  const freshness = snapshot.freshness_overview || {};
  const candidateCount = overview.candidates ?? snapshot.summary_counts.candidates ?? 0;
  const newBuyCount = overview.new_buys ?? snapshot.summary_counts.buys ?? 0;
  const cards = [
    { label: "Net liquidation", value: fmtCurrency(portfolio.net_liquidation_usd) },
    { label: "Settled cash", value: fmtCurrency(portfolio.settled_cash_usd) },
    { label: "Available cash", value: fmtCurrency(portfolio.available_cash_usd) },
    { label: "Positions", value: portfolio.position_count },
    { label: "New Buys", value: newBuyCount },
    { label: "Candidates", value: candidateCount },
    { label: "SELL", value: overview.sells ?? snapshot.summary_counts.sells },
    { label: "REVIEW", value: overview.reviews ?? snapshot.summary_counts.reviews },
  ];
  const sectorRows = Object.entries(portfolio.sector_weights || {})
    .map(([name, pct]) => `<tr><td>${escapeHtml(name)}</td><td>${fmtPct(pct)}</td></tr>`)
    .join("");
  const exchangeRows = Object.entries(portfolio.exchange_weights || {})
    .map(([name, pct]) => `<tr><td>${escapeHtml(name)}</td><td>${fmtPct(pct)}</td></tr>`)
    .join("");
  const sectorTable = sectorRows || "<tr><td colspan='2' class='muted'>No live portfolio positions loaded.</td></tr>";
  const exchangeTable = exchangeRows || "<tr><td colspan='2' class='muted'>No live portfolio positions loaded.</td></tr>";
  const healthFlags = (snapshot.health_flags || [])
    .map((flag) => `<li>${escapeHtml(flag)}</li>`)
    .join("");
  const modeNote = snapshot.read_only
    ? `<p class="muted">Read-only mode: portfolio balances, positions, and live orders stay empty until live IBKR mode is enabled.</p>`
    : "";
  const portfolioRealityNote =
    snapshot.read_only && portfolio.position_count === 0
      ? "<p class='muted'><strong>This is not your live IBKR portfolio.</strong> Read-only mode skips the broker portfolio pull entirely, so zero balances and zero positions are expected here.</p>"
      : "";
  const candidateNote =
    overview.is_candidate_heavy
      ? `<p class="muted">This snapshot is candidate-heavy, not portfolio-heavy. It contains ${candidateCount} off-watchlist candidate${candidateCount === 1 ? "" : "s"} and ${newBuyCount} watchlist buy${newBuyCount === 1 ? "" : "s"}.</p>`
      : "";
  return `
    ${renderCards(cards)}
    ${modeNote}
    ${portfolioRealityNote}
    ${candidateNote}
    <section class="summary-grid">
      ${renderInlineMetrics("Freshness At A Glance", [
        { label: "Blocking now", value: freshness.blocking_now ?? 0 },
        { label: "Stale in queue", value: freshness.stale_in_queue ?? 0 },
        { label: "Due soon", value: freshness.due_soon ?? 0 },
        { label: "Fresh count", value: freshness.fresh_count ?? 0 },
      ])}
      ${renderInlineMetrics("Cash Snapshot", [
        { label: "Total cash", value: fmtCurrency(cashSummary.total_cash_usd) },
        { label: "Unsettled", value: fmtCurrency(cashSummary.unsettled_cash_usd) },
        { label: "Buffer reserve", value: fmtCurrency(cashSummary.buffer_reserve_usd) },
        {
          label: "Pending inflows",
          value: fmtCurrency(cashSummary.pending_inflows_total_usd),
        },
      ])}
    </section>
    ${renderCandidatePreview("Candidate Preview", snapshot.actions.watchlist_candidate)}
    ${renderCandidatePreview("Watchlist Buys Ready For Review", snapshot.actions.watchlist_buy)}
    <section>
      <h3 class="section-title">Portfolio Health</h3>
      <ul>${healthFlags || "<li class='muted'>No portfolio health flags.</li>"}</ul>
    </section>
    <section>
      <h3 class="section-title">Cash Timeline</h3>
      ${renderCashTimelineTable(snapshot.cash_timeline, "No pending inflows.")}
    </section>
    <section class="cards">
      <article class="card">
        <h3>Sector Concentration</h3>
        <table><tbody>${sectorTable}</tbody></table>
      </article>
      <article class="card">
        <h3>Exchange Concentration</h3>
        <table><tbody>${exchangeTable}</tbody></table>
      </article>
    </section>
  `;
}

function renderActions() {
  const actions = state.snapshot.actions;
  const actionSections = [
    actions.sell_stop_breach,
    actions.sell_hard,
    actions.sell_soft_review,
    actions.review_macro,
    actions.review_stop_breach,
    actions.add,
    actions.trim,
    actions.review,
    actions.dip_watch,
    actions.hold,
  ];
  if (actionSections.every((items) => !items || items.length === 0)) {
    return `
      <section>
        <h3 class="section-title">Held-Position Actions</h3>
        <p class="muted">No held-position actions are present in this snapshot. If this is a read-only or candidate-only screen, the useful names are in Watchlist & Candidates.</p>
      </section>
      ${renderCandidatePreview("Candidate Preview", actions.watchlist_candidate)}
      ${renderCandidatePreview("Watchlist Buys Ready For Review", actions.watchlist_buy)}
    `;
  }
  return `
    ${renderActionTable("Stop Breaches", actions.sell_stop_breach, [
      { label: "Price", render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Settles", render: (item) => escapeHtml(item.settlement_date || "—") },
    ])}
    ${renderActionTable("Fundamental Sells", actions.sell_hard, [
      { label: "Price", render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Settles", render: (item) => escapeHtml(item.settlement_date || "—") },
    ])}
    ${renderActionTable("Soft Rejections", actions.sell_soft_review, [
      { label: "Health", render: (item) => escapeHtml(item.analysis?.health_adj ?? "—") },
      { label: "Growth", render: (item) => escapeHtml(item.analysis?.growth_adj ?? "—") },
    ])}
    ${renderActionTable("Macro Reviews", actions.review_macro, [
      { label: "Health", render: (item) => escapeHtml(item.analysis?.health_adj ?? "—") },
      { label: "Growth", render: (item) => escapeHtml(item.analysis?.growth_adj ?? "—") },
    ])}
    ${renderActionTable("Stop Breaches Under Review", actions.review_stop_breach, [
      { label: "Health", render: (item) => escapeHtml(item.analysis?.health_adj ?? "—") },
      { label: "Growth", render: (item) => escapeHtml(item.analysis?.growth_adj ?? "—") },
    ])}
    ${renderActionTable("Adds", actions.add, [
      { label: "Price", render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Cost", render: (item) => fmtCurrency(Math.abs(item.cash_impact_usd ?? 0)) },
    ])}
    ${renderActionTable("Trims", actions.trim, [
      { label: "Price", render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Settles", render: (item) => escapeHtml(item.settlement_date || "—") },
    ])}
    ${renderActionTable("Review Queue", actions.review, [
      { label: "Health", render: (item) => escapeHtml(item.analysis?.health_adj ?? "—") },
      { label: "Growth", render: (item) => escapeHtml(item.analysis?.growth_adj ?? "—") },
    ])}
    ${renderDipWatch(actions.dip_watch)}
    ${renderActionTable("Holds", actions.hold, [
      {
        label: "Entry",
        render: (item) =>
          fmtLocalMoney(item.analysis?.entry_price, item.analysis?.currency),
      },
      {
        label: "Current",
        render: (item) =>
          fmtLocalMoney(item.position?.current_price_local, item.position?.currency),
      },
    ])}
  `;
}

function renderDipWatch(items) {
  if (!items || !items.length) {
    return `<section><h3 class="section-title">Dip Watch</h3><p class="muted">No dip-watch candidates.</p></section>`;
  }
  const rows = items
    .map(
      (item) => `
      <tr>
        <td>${escapeHtml(item.stars)}</td>
        <td><button type="button" class="ticker-link" data-ticker="${escapeHtml(item.ticker_yf)}">${escapeHtml(item.ticker_ibkr)}</button></td>
        <td>${fmtNumber(item.score, 1)}</td>
        <td>${fmtPct(item.dip_pct)}</td>
        <td>${item.risk_reward ?? "—"}</td>
        <td>${escapeHtml(item.run_ticker)}</td>
      </tr>
    `,
    )
    .join("");
  return `
    <section>
      <h3 class="section-title">Dip Watch</h3>
      <table>
        <thead><tr><th>Stars</th><th>Ticker</th><th>Score</th><th>Dip</th><th>R/R</th><th>Run Ticker</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function renderWatchlist() {
  const actions = state.snapshot.actions;
  return `
    ${renderActionTable("New Buys", actions.watchlist_buy, [
      { label: "Price", render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Cost", render: (item) => fmtCurrency(Math.abs(item.cash_impact_usd ?? 0)) },
    ])}
    ${renderActionTable("Watchlist Candidates", actions.watchlist_candidate, [
      { label: "Health", render: (item) => escapeHtml(item.analysis?.health_adj ?? "—") },
      { label: "Growth", render: (item) => escapeHtml(item.analysis?.growth_adj ?? "—") },
    ])}
    ${renderActionTable("Watchlist Monitoring", actions.watchlist_monitor)}
    ${renderActionTable("Watchlist Remove", actions.watchlist_remove)}
  `;
}

function renderOrders() {
  const orders = state.snapshot.orders || [];
  const cashSummary = state.snapshot.cash_summary || {};
  const rows = orders
    .map(
      (order) => `
        <tr>
          <td>${escapeHtml(order.ticker || order.symbol || "—")}</td>
          <td>${escapeHtml(order.side || "—")}</td>
          <td>${escapeHtml(order.orderType || "—")}</td>
          <td>${escapeHtml(order.status || "—")}</td>
          <td>${escapeHtml(order.remainingSize || order.totalSize || "—")}</td>
        </tr>
      `,
    )
    .join("");
  return `
    ${renderCards([
      { label: "Settled cash", value: fmtCurrency(cashSummary.settled_cash_usd) },
      { label: "Available cash", value: fmtCurrency(cashSummary.available_cash_usd) },
      { label: "Buffer reserve", value: fmtCurrency(cashSummary.buffer_reserve_usd) },
      { label: "Unsettled cash", value: fmtCurrency(cashSummary.unsettled_cash_usd) },
    ])}
    ${
      state.snapshot.read_only
        ? "<p class='muted'>Read-only mode: live orders and live cash context require IBKR_DASHBOARD_READ_ONLY=false.</p>"
        : ""
    }
    <section>
      <h3 class="section-title">Cash Plan</h3>
      <div class="summary-grid">
        ${renderInlineMetrics("Spendable Today", [
          { label: "Settled cash", value: fmtCurrency(cashSummary.settled_cash_usd) },
          {
            label: "After recommended buys",
            value: fmtCurrency(cashSummary.settled_cash_after_recommended_buys_usd),
          },
        ])}
        ${renderInlineMetrics("Pending Inflows", [
          {
            label: "Total pending",
            value: fmtCurrency(cashSummary.pending_inflows_total_usd),
          },
          {
            label: "Next settlement",
            value: cashSummary.next_settlement_date || "—",
          },
        ])}
      </div>
      ${renderCashTimelineTable(
        cashSummary.pending_inflows || [],
        "No pending inflows.",
      )}
    </section>
    <section>
      <h3 class="section-title">Live Orders</h3>
      <table>
        <thead><tr><th>Ticker</th><th>Side</th><th>Type</th><th>Status</th><th>Remaining</th></tr></thead>
        <tbody>${rows || "<tr><td colspan='5' class='muted'>No live orders.</td></tr>"}</tbody>
      </table>
    </section>
  `;
}

function renderRefresh() {
  const freshness = state.snapshot.freshness;
  const staleEligible =
    freshness.blocking_now.length + freshness.stale_in_queue.length;
  const dueSoonEligible = freshness.due_soon.length;
  const allFresh =
    staleEligible === 0 &&
    dueSoonEligible === 0 &&
    freshness.candidate_blocked.length === 0 &&
    freshness.fresh_count > 0;
  const explainer = allFresh
    ? "All loaded analyses are fresh. Use a ticker list if you want to force a rerun."
    : "These counts describe the current snapshot. Refresh jobs appear here only after you queue one.";
  return `
    ${renderCards([
      { label: "Blocking now", value: freshness.blocking_now.length },
      { label: "Stale in queue", value: freshness.stale_in_queue.length },
      { label: "Due soon", value: freshness.due_soon.length },
      { label: "Fresh count", value: freshness.fresh_count },
    ])}
    <section>
      <h3 class="section-title">Create Refresh Job</h3>
      <p class="muted">${escapeHtml(explainer)}</p>
      <div class="jobs-controls">
        <button id="job-stale" type="button" ${staleEligible === 0 ? "disabled" : ""}>Queue stale positions (${staleEligible})</button>
        <button id="job-due-soon" type="button" ${dueSoonEligible === 0 ? "disabled" : ""}>Queue due soon (${dueSoonEligible})</button>
        <input id="job-ticker-input" type="text" placeholder="7203.T, MEGP.L">
        <button id="job-custom" type="button">Queue ticker list</button>
      </div>
    </section>
    <section>
      <h3 class="section-title">Refresh Jobs</h3>
      ${renderJobsTable()}
    </section>
  `;
}

function renderJobsTable() {
  if (!state.jobs.length) {
    return "<p class='muted'>No refresh jobs yet. Queue a job above to start one.</p>";
  }
  const rows = state.jobs
    .map(
      (job) => `
      <tr>
        <td title="${escapeHtml(job.job_id)}">${escapeHtml((job.job_id || "").slice(0, 8) || "—")}</td>
        <td>${escapeHtml(job.scope)}</td>
        <td>${escapeHtml(job.status)}</td>
        <td>${escapeHtml(job.created_at)}</td>
        <td>${escapeHtml(job.finished_at || "—")}</td>
      </tr>
    `,
    )
    .join("");
  return `<table><thead><tr><th>Job</th><th>Scope</th><th>Status</th><th>Created</th><th>Finished</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderSettings() {
  const settings = state.settings || {};
  const modeValue = settings.read_only ? "true" : "false";
  return `
    <section>
      <h3 class="section-title">Snapshot Settings</h3>
      <p class="muted">These settings control the next snapshot load. Use startup flags when you want a one-off session override.</p>
      <form id="settings-form" class="settings-form">
        <label>IBKR account ID<input name="account_id" value="${escapeHtml(settings.account_id || "")}" placeholder="U20958465"></label>
        <label>Watchlist name<input name="watchlist_name" value="${escapeHtml(settings.watchlist_name || "")}"></label>
        <label>Snapshot mode
          <select name="read_only">
            <option value="false" ${modeValue === "false" ? "selected" : ""}>Live IBKR portfolio</option>
            <option value="true" ${modeValue === "true" ? "selected" : ""}>Read-only results only</option>
          </select>
        </label>
        <label>Max age days<input name="max_age_days" type="number" value="${escapeHtml(settings.max_age_days ?? 14)}"></label>
        <label>Refresh limit<input name="refresh_limit" type="number" value="${escapeHtml(settings.refresh_limit ?? 10)}"></label>
        <label>Quick mode default
          <select name="quick_mode_default">
            <option value="true" ${settings.quick_mode_default ? "selected" : ""}>true</option>
            <option value="false" ${settings.quick_mode_default === false ? "selected" : ""}>false</option>
          </select>
        </label>
        <label>Notes<textarea name="notes">${escapeHtml(settings.notes || "")}</textarea></label>
        <button type="submit">Save settings</button>
      </form>
    </section>
  `;
}

function renderDetailSection(title, rows) {
  const filteredRows = rows.filter((row) => row.value !== undefined);
  if (!filteredRows.length) {
    return "";
  }
  return `
    <section class="detail-section">
      <h4>${escapeHtml(title)}</h4>
      <dl class="detail-list">
        ${filteredRows
          .map(
            (row) => `
              <dt>${escapeHtml(row.label)}</dt>
              <dd>${formatDetailValue(row.value)}</dd>
            `,
          )
          .join("")}
      </dl>
    </section>
  `;
}

function renderStructuredSections(structured) {
  if (!structured || !Object.keys(structured).length) {
    return "";
  }
  const sections = [
    ["Prediction Snapshot", structured.prediction_snapshot],
    ["Final Decision", structured.final_decision],
    ["Investment Analysis", structured.investment_analysis],
    ["Risk Analysis", structured.risk_analysis],
    ["Artifact Statuses", structured.artifact_statuses],
    ["Analysis Validity", structured.analysis_validity],
  ]
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([title, value]) => {
      const rows =
        typeof value === "object" && !Array.isArray(value)
          ? Object.entries(value).map(([label, entry]) => ({ label, value: entry }))
          : [{ label: title, value }];
      return renderDetailSection(title, rows);
    })
    .join("");
  return sections ? `<div class="detail-grid">${sections}</div>` : "";
}

function renderDrilldown(payload) {
  const position = payload.position || {};
  const analysis = payload.analysis || {};
  const tradeBlock = analysis.trade_block || {};

  return `
    <h3>${escapeHtml(payload.ticker_ibkr)}</h3>
    <div class="detail-grid">
      ${renderDetailSection("Holding", [
        { label: "Action", value: payload.action },
        { label: "Reason", value: payload.reason },
        { label: "Urgency", value: payload.urgency },
        { label: "Quantity", value: position.quantity },
        { label: "Live order note", value: payload.live_order_note },
        {
          label: "Avg cost",
          value:
            position.avg_cost_local !== undefined
              ? fmtLocalMoney(position.avg_cost_local, position.currency)
              : undefined,
        },
        {
          label: "Current price",
          value:
            position.current_price_local !== undefined
              ? fmtLocalMoney(position.current_price_local, position.currency)
              : undefined,
        },
        { label: "Market value", value: fmtCurrency(position.market_value_usd) },
        { label: "Unrealized P/L", value: fmtCurrency(position.unrealized_pnl_usd) },
      ])}
      ${renderDetailSection("Latest Analysis", [
        { label: "Verdict", value: analysis.verdict },
        { label: "Date", value: analysis.analysis_date },
        { label: "Age (days)", value: analysis.age_days },
        { label: "Health", value: analysis.health_adj },
        { label: "Growth", value: analysis.growth_adj },
        { label: "Zone", value: analysis.zone },
        { label: "Conviction", value: analysis.conviction },
        { label: "Quick mode", value: analysis.is_quick_mode },
      ])}
      ${renderDetailSection("Trade Thesis", [
        {
          label: "Entry",
          value:
            analysis.entry_price !== undefined
              ? fmtLocalMoney(analysis.entry_price, analysis.currency)
              : undefined,
        },
        {
          label: "Stop",
          value:
            analysis.stop_price !== undefined
              ? fmtLocalMoney(analysis.stop_price, analysis.currency)
              : undefined,
        },
        {
          label: "Target 1",
          value:
            analysis.target_1_price !== undefined
              ? fmtLocalMoney(analysis.target_1_price, analysis.currency)
              : undefined,
        },
        {
          label: "Target 2",
          value:
            analysis.target_2_price !== undefined
              ? fmtLocalMoney(analysis.target_2_price, analysis.currency)
              : undefined,
        },
        { label: "Trade action", value: tradeBlock.action },
        { label: "Target size %", value: tradeBlock.size_pct },
        { label: "Risk/Reward", value: tradeBlock.risk_reward },
      ])}
    </div>
    ${renderStructuredSections(payload.structured)}
    ${
      payload.report_markdown_html
        ? `<section class="detail-section"><h4>Report</h4><div class="markdown-body">${payload.report_markdown_html}</div></section>`
        : ""
    }
    ${
      payload.article_markdown_html
        ? `<section class="detail-section"><h4>Article</h4><div class="markdown-body">${payload.article_markdown_html}</div></section>`
        : ""
    }
    ${payload.note ? `<p class="muted">${escapeHtml(payload.note)}</p>` : ""}
  `;
}

function renderActiveTab() {
  if (!state.snapshot && state.activeTab !== "settings") {
    const message =
      state.snapshotMeta.status === "loading"
        ? "Snapshot loading in background…"
        : state.snapshotMeta.status === "error"
          ? "Snapshot unavailable. Use Refresh Snapshot to retry."
          : "Snapshot not loaded yet. It should load automatically in a moment.";
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
        "<p class='muted'>Snapshot is still loading. Try again in a moment.</p>";
      return;
    }
    elements.drilldown().innerHTML = renderDrilldown(payload);
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
  const headline = macro.headline ? `Headline: ${macro.headline}` : "Macro event detected.";
  alert.innerHTML = `<strong>Macro alert:</strong> ${escapeHtml(headline)} (${escapeHtml(String(macro.correlation_pct || "—"))}% of held positions)`;
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
  alert.classList.remove("hidden");
  alert.innerHTML =
    "<strong>Read-only snapshot:</strong> this dashboard is showing saved analysis results from <code>results/</code>, not your live IBKR portfolio. Switch Snapshot mode to live in Settings, or restart with <code>--live</code> / <code>IBKR_DASHBOARD_READ_ONLY=false</code> and working broker credentials."
    + accountHint;
}

function updateStatus() {
  const status = elements.status();
  const context = elements.context();
  if (state.snapshotMeta.status === "loading") {
    status.textContent = "Snapshot loading…";
    context.textContent = "";
    return;
  }
  if (state.snapshotMeta.status === "error") {
    status.textContent = "Snapshot load failed";
    context.textContent = "";
    return;
  }
  if (!state.snapshot) {
    status.textContent = "No snapshot loaded";
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
  status.textContent = `Snapshot ${source} at ${state.snapshot.as_of} (${freshness}, ${mode})`;
  const parts = [];
  if (state.snapshot.portfolio?.account_id) {
    parts.push(`Account ${state.snapshot.portfolio.account_id}`);
  }
  if (state.snapshot.watchlist?.name) {
    parts.push(`Watchlist ${state.snapshot.watchlist.name}`);
  }
  parts.push(state.snapshot.read_only ? "Read-only results snapshot" : "Live IBKR snapshot");
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

function bindDynamicHandlers() {
  document.querySelectorAll(".ticker-link[data-ticker]").forEach((button) => {
    button.addEventListener("click", () => loadDrilldown(button.dataset.ticker));
  });

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
}

async function initializeDashboard() {
  bindStaticHandlers();
  await loadSettings();
  await loadPortfolio(false);
  syncJobsPolling();
}

initializeDashboard();
