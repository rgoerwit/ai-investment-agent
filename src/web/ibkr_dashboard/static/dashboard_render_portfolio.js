// Hold-row helpers mirror the CLI HOLDS renderer
// (src/ibkr/portfolio_report_positions.py): entry falls back from the analysis
// entry price to the position's average cost, and gain/weight are derived the
// same way so the dashboard and the printed report agree.
function holdEntry(item) {
  if (item.analysis?.entry_price) return item.analysis.entry_price;
  if (item.position?.avg_cost_local) return item.position.avg_cost_local;
  return null;
}

function holdEntryLabel(item) {
  return item.analysis?.entry_price
    ? "Entry price from analysis"
    : "Cost basis (average cost) — no analysis entry price";
}

function holdCurrency(item) {
  return item.analysis?.currency || item.position?.currency;
}

function holdWeight(item) {
  const marketValue = item.position?.market_value_usd;
  const nav = state.snapshot?.portfolio?.net_liquidation_usd;
  if (!marketValue || !nav || nav <= 0) return "—";
  return `${((marketValue / nav) * 100).toFixed(1)}%`;
}

function holdGain(item) {
  const entry = holdEntry(item);
  const current = item.position?.current_price_local;
  if (!entry || !current) return "—";
  return fmtPct(((current - entry) / entry) * 100);
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
      ? `<p class="muted">This view is candidate-heavy, not portfolio-heavy. It contains ${candidateCount} off-watchlist candidate${candidateCount === 1 ? "" : "s"} and ${newBuyCount} watchlist buy${newBuyCount === 1 ? "" : "s"}.</p>`
      : "";
  return `
    ${renderCards(cards)}
    ${modeNote}
    ${portfolioRealityNote}
    ${candidateNote}
    <section class="summary-grid">
      ${renderInlineMetrics("Freshness At A Glance", [
        { label: "Needs review", value: freshness.blocking_now ?? 0 },
        { label: "Refresh queue", value: freshness.stale_in_queue ?? 0 },
        { label: "Needs full refresh", value: freshness.candidate_blocked ?? 0 },
        { label: "Due soon", value: freshness.due_soon ?? 0 },
        { label: "Fresh", value: freshness.fresh_count ?? 0 },
      ])}
      ${renderInlineMetrics("Cash Overview", [
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
      ${renderConcentrationCard(
        "sector",
        "Sector Concentration",
        "Sector",
        portfolio.sector_weights,
        "No live portfolio positions loaded.",
        snapshot.concentration_limits?.sector,
      )}
      ${renderConcentrationCard(
        "exchange",
        "Exchange Concentration",
        "Exchange",
        portfolio.exchange_weights,
        "No live portfolio positions loaded.",
        snapshot.concentration_limits?.exchange,
      )}
    </section>
  `;
}

function renderActions() {
  const actions = state.snapshot.actions;
  const canonicalSections = actions.action_sections || [];
  const fallbackSellItems = [
    ...(actions.sell_stop_breach || []),
    ...(actions.sell_hard || []),
    ...(actions.sell_profit_take || []),
    ...(actions.sell_soft_review || []),
  ];
  const fallbackSections = [
    { key: "sell_recommendations", title: "Sell Recommendations", kind: "reconciliation_items", items: fallbackSellItems },
    { key: "sell_related_reviews", title: "Sell-Related Reviews", kind: "reconciliation_items", items: [...(actions.review_stop_breach || []), ...(actions.review_macro || []), ...(actions.review_profit_take || [])] },
    { key: "add", title: "Adds", kind: "reconciliation_items", items: actions.add || [] },
    { key: "trim", title: "Trims", kind: "reconciliation_items", items: actions.trim || [] },
    { key: "review", title: "Review Queue", kind: "reconciliation_items", items: actions.review || [] },
    { key: "dip_watch", title: "Dip Watch", kind: "dip_watch", items: actions.dip_watch || [] },
    { key: "hold", title: "Holds", kind: "reconciliation_items", items: actions.hold || [] },
  ];
  const actionSections = (canonicalSections.length ? canonicalSections : fallbackSections).filter(
    (section) => section.items && section.items.length > 0,
  );
  if (actionSections.length === 0) {
    return `
      <section>
        <h3 class="section-title">Held-Position Actions</h3>
        <p class="muted">No held-position actions are present in the current data. If this is a read-only or candidate-only screen, the useful names are in Watchlist & Candidates.</p>
      </section>
      ${renderCandidatePreview("Candidate Preview", actions.watchlist_candidate)}
      ${renderCandidatePreview("Watchlist Buys Ready For Review", actions.watchlist_buy)}
    `;
  }
  return actionSections
    .map((section) => {
      if (section.key === "sell_recommendations") {
        return renderActionTable(section.title, section.items, [
          { label: "Type", render: (item) => escapeHtml(sellTypeLabel(item)) },
          { label: "Price", numeric: true, render: (item) => fmtNumber(item.suggested_price, 2) },
          {
            label: "Would Settle",
            render: (item) => escapeHtml(item.settlement_date || "—"),
          },
          { label: "Profit-Take Detail", render: profitTakeDetail },
        ]);
      }
      if (section.key === "sell_related_reviews") {
        return renderActionTable(section.title, section.items, [
          { label: "Type", render: (item) => escapeHtml(sellTypeLabel(item)) },
          { label: "Health", numeric: true, render: (item) => fmtScorePct(item.analysis?.health_adj) },
          { label: "Growth", numeric: true, render: (item) => fmtScorePct(item.analysis?.growth_adj) },
          { label: "Profit-Take Detail", render: profitTakeDetail },
        ]);
      }
      if (section.key === "add") {
        return renderActionTable(section.title, section.items, [
          { label: "Price", numeric: true, render: (item) => fmtNumber(item.suggested_price, 2) },
          { label: "Cost", numeric: true, render: (item) => fmtCurrency(Math.abs(item.cash_impact_usd ?? 0)) },
        ]);
      }
      if (section.key === "trim") {
        return renderActionTable(section.title, section.items, [
          { label: "Price", numeric: true, render: (item) => fmtNumber(item.suggested_price, 2) },
          {
            label: "Would Settle",
            render: (item) => escapeHtml(item.settlement_date || "—"),
          },
        ]);
      }
      if (section.key === "review") {
        return renderActionTable(section.title, section.items, [
          { label: "Health", numeric: true, render: (item) => fmtScorePct(item.analysis?.health_adj) },
          { label: "Growth", numeric: true, render: (item) => fmtScorePct(item.analysis?.growth_adj) },
        ]);
      }
      if (section.key === "dip_watch") {
        return renderDipWatch(section.items);
      }
      if (section.key === "hold") {
        return renderActionTable(
          section.title,
          section.items,
          [
            { label: "Weight", numeric: true, render: holdWeight },
            {
              label: "Entry",
              numeric: true,
              render: (item) => fmtLocalMoney(holdEntry(item), holdCurrency(item)),
              title: holdEntryLabel,
            },
            {
              label: "Current",
              numeric: true,
              render: (item) =>
                fmtLocalMoney(item.position?.current_price_local, item.position?.currency),
            },
            { label: "Gain %", numeric: true, render: holdGain },
            {
              label: "Stop",
              numeric: true,
              render: (item) => fmtLocalMoney(item.analysis?.stop_price, holdCurrency(item)),
            },
            {
              label: "Target",
              numeric: true,
              render: (item) =>
                fmtLocalMoney(item.analysis?.target_1_price, holdCurrency(item)),
            },
          ],
          { omitReason: true },
        );
      }
      return renderActionTable(section.title, section.items, []);
    })
    .join("");
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
        <td class="num">${fmtNumber(item.score, 1)}</td>
        <td class="num">${fmtPct(item.dip_pct)}</td>
        <td class="num">${escapeHtml(item.risk_reward ?? "—")}</td>
        <td>${escapeHtml(item.run_ticker)}</td>
      </tr>
    `,
    )
    .join("");
  return `
    <section>
      <h3 class="section-title">Dip Watch</h3>
      <table>
        <thead><tr><th>Stars</th><th>Ticker</th><th class="num">Score</th><th class="num">Dip</th><th class="num">R/R</th><th>Run Ticker</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function renderLoadedWatchlist(watchlist) {
  const tickers = watchlist?.tickers || [];
  const status = watchlist?.status || (watchlist?.total == null ? "not_loaded" : "loaded");
  if (status === "unavailable") {
    return `
      <section>
        <h3 class="section-title">IBKR Watchlist Unavailable</h3>
        <p class="muted">The brokerage watchlist could not be read. Membership is unknown; candidate rows below are advisory and no removals are proposed.</p>
      </section>
    `;
  }
  if (status === "not_loaded") {
    return `
      <section>
        <h3 class="section-title">No IBKR Watchlist Loaded</h3>
        <p class="muted">Candidate rows below are advisory. Watchlist membership has not been verified.</p>
      </section>
    `;
  }
  const title = watchlist?.name
    ? `Loaded IBKR Watchlist: ${watchlist.name}`
    : "Loaded IBKR Watchlist";
  const subtitle =
    watchlist?.total !== null && watchlist?.total !== undefined
      ? `<p class="muted">${watchlist.total} ticker${watchlist.total === 1 ? "" : "s"} loaded from IBKR for the current view.</p>`
      : "";
  if (!tickers.length) {
    return `
      <section>
        <h3 class="section-title">${escapeHtml(title)}</h3>
        ${subtitle}
        <p class="muted">No tickers were loaded from the named IBKR watchlist.</p>
      </section>
    `;
  }
  const rows = tickers
    .map(
      (ticker) => `
        <tr>
          <td>${escapeHtml(ticker)}</td>
        </tr>
      `,
    )
    .join("");
  return `
    <section>
      <h3 class="section-title">${escapeHtml(title)}</h3>
      ${subtitle}
      <table>
        <thead><tr><th>Ticker</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

// One short line per breached dimension: "exchange T 52.7% > 40%".
function breachLines(breaches) {
  return breaches
    .map((breach) =>
      escapeHtml(
        `${breach.dimension} ${breach.key} ` +
          `${Number(breach.projected_pct).toFixed(1)}% > ${Number(breach.limit_pct).toFixed(0)}%`,
      ),
    )
    .join("<br>");
}

// "Why" cell: prefer the structured breach list, fall back to the pre-joined
// concentration string (or a plain removal reason) for back-compat.
function breachWhy(item) {
  if (Array.isArray(item.breaches) && item.breaches.length) {
    return breachLines(item.breaches);
  }
  return escapeHtml(item.concentration || item.removal_reason || item.reason || "—");
}

// Grouping label without per-candidate projections — mirrors _breach_category
// (src/ibkr/portfolio_report.py) so items with the same overweight bucket collapse.
function breachCategory(breaches) {
  return breaches
    .map((breach) => `${breach.dimension} ${breach.key} > ${Number(breach.limit_pct).toFixed(0)}%`)
    .join(" + ");
}

// Per-group display label, keeping the worst projection per dimension so the
// magnitude survives grouping (mirrors the CLI withheld-by-concentration footer).
function breachGroupLabel(notes) {
  return notes[0].breaches
    .map((breach, dim) => {
      const worst = Math.max(
        ...notes.map((note) => Number(note.breaches[dim].projected_pct)),
      );
      return (
        `${breach.dimension} ${breach.key} up to ` +
        `${worst.toFixed(1)}% > ${Number(breach.limit_pct).toFixed(0)}%`
      );
    })
    .join(" + ");
}

function renderWithheldGrouped(items) {
  if (!items || !items.length) return "";
  const groups = new Map();
  for (const item of items) {
    const breaches = Array.isArray(item.breaches) ? item.breaches : [];
    const key = breaches.length
      ? breachCategory(breaches)
      : item.concentration || item.reason || "—";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const rows = [...groups.values()]
    .map((notes) => {
      const label = notes[0].breaches?.length
        ? breachGroupLabel(notes)
        : notes[0].concentration || notes[0].reason || "—";
      const tickers = notes
        .map((note) => escapeHtml(note.ticker_ibkr || note.ticker_yf || "—"))
        .join(", ");
      return `<tr><td>${escapeHtml(label)}</td><td>${tickers}</td></tr>`;
    })
    .join("");
  return `
    <section>
      <h3 class="section-title">Withheld By Concentration</h3>
      <table>
        <thead><tr><th>Overweight bucket</th><th>Withheld tickers</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function renderWatchlist() {
  const actions = state.snapshot.actions;
  const watchlist = state.snapshot.watchlist || {};
  return `
    ${renderLoadedWatchlist(watchlist)}
    ${renderActionTable("New Buys", actions.watchlist_buy, [
      { label: "Price", numeric: true, render: (item) => fmtNumber(item.suggested_price, 2) },
      { label: "Cost", numeric: true, render: fmtBuyCost },
    ])}
    ${renderActionTable("Watchlist Candidates", actions.watchlist_candidate, [
      { label: "Health", numeric: true, render: (item) => fmtScorePct(item.analysis?.health_adj) },
      { label: "Growth", numeric: true, render: (item) => fmtScorePct(item.analysis?.growth_adj) },
    ])}
    ${(actions.watchlist_in_flight || []).length
      ? renderActionTable("BUY Orders Already In Flight", actions.watchlist_in_flight, [
          {
            label: "Watchlist Status",
            render: (item) =>
              item.watchlist_membership === "not_on_loaded_watchlist"
                ? "Not on loaded watchlist"
                : "Membership unknown",
          },
        ])
      : ""}
    ${renderActionTable("Watchlist Monitoring", actions.watchlist_monitor)}
    ${renderActionTable("Watchlist Remove", actions.watchlist_remove, [
      { label: "Why", render: breachWhy },
    ])}
    ${renderWithheldGrouped(actions.watchlist_withheld)}
  `;
}

function renderOrders() {
  const orders = state.snapshot.orders || [];
  const cashSummary = state.snapshot.cash_summary || {};
  const immediateBuyCost = Number(cashSummary.recommended_buy_cost_usd || 0);
  const spendableRows = [
    { label: "Settled cash", value: fmtCurrency(cashSummary.settled_cash_usd) },
    {
      label: "Immediate ADD / BUY cost",
      value: immediateBuyCost > 0 ? fmtCurrency(immediateBuyCost) : "None queued",
    },
  ];
  if (immediateBuyCost > 0) {
    spendableRows.push({
      label: "After current ADD / BUY actions",
      value: fmtCurrency(cashSummary.settled_cash_after_recommended_buys_usd),
    });
  }
  const buyWorkflowNote =
    immediateBuyCost > 0
      ? "<p class='muted'>Only immediate ADD and watchlist BUY actions reserve cash here.</p>"
      : "<p class='muted'>Dip-watch ideas and off-watchlist candidates do not reserve cash here. Cash only moves once a name becomes an ADD or a watchlist BUY.</p>";
  const ordersError = state.snapshot.errors?.live_orders;
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
  // A live-orders fetch failure must not look like "no open orders" — that would
  // invite a duplicate order. Surface the degraded state explicitly.
  const ordersEmptyRow = ordersError
    ? "<tr><td colspan='5' class='muted'>Live-order data could not be loaded.</td></tr>"
    : "<tr><td colspan='5' class='muted'>No live orders.</td></tr>";
  const ordersErrorBanner = ordersError
    ? `<p class="error">Live orders unavailable — open-order dedup is disabled; verify open orders directly in IBKR before placing new ones. (${escapeHtml(ordersError)})</p>`
    : "";
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
        ${renderInlineMetrics("Spendable Today", spendableRows)}
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
      ${buyWorkflowNote}
      ${renderCashTimelineTable(
        cashSummary.pending_inflows || [],
        "No pending inflows.",
      )}
    </section>
    <section>
      <h3 class="section-title">Live Orders</h3>
      ${ordersErrorBanner}
      <table>
        <thead><tr><th>Ticker</th><th>Side</th><th>Type</th><th>Status</th><th>Remaining</th></tr></thead>
        <tbody>${rows || ordersEmptyRow}</tbody>
      </table>
    </section>
  `;
}
