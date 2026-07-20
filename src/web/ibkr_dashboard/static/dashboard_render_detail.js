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

function renderReportActions(payload) {
  const actions = [];
  const structured = payload.structured || {};
  const hasDecisionDetail = hasStructuredContent(structured, [
    "prediction_snapshot",
    "final_decision",
    "investment_analysis",
    "risk_analysis",
  ]);
  const hasAgentOutputs = hasStructuredContent(structured, [
    "reports",
    "artifact_statuses",
    "analysis_validity",
  ]);
  if (hasDecisionDetail) {
    actions.push(
      '<button type="button" class="report-open" data-report-kind="decision">Decision Detail</button>',
    );
  }
  if (hasAgentOutputs) {
    actions.push(
      '<button type="button" class="report-open" data-report-kind="diagnostics">Agent Outputs</button>',
    );
  }
  if (payload.report_markdown_html) {
    actions.push(
      '<button type="button" class="report-open" data-report-kind="report">Open Report</button>',
    );
  }
  if (payload.article_markdown_html) {
    actions.push(
      '<button type="button" class="report-open" data-report-kind="article">Open Article</button>',
    );
  }
  if (!actions.length) {
    return payload.note ? `<p class="muted">${escapeHtml(payload.note)}</p>` : "";
  }
  return `
    <section class="detail-section report-actions">
      <h4>Full Report</h4>
      <div class="topbar-actions report-action-buttons">${actions.join("")}</div>
    </section>
  `;
}

function hasStructuredContent(structured, keys) {
  return keys.some((key) => {
    const value = structured?.[key];
    if (value === null || value === undefined) return false;
    if (typeof value === "string") return value.trim().length > 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === "object") return Object.keys(value).length > 0;
    return true;
  });
}

function titleFromKey(key) {
  return String(key || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function renderStructuredDocument(title, structured, sections) {
  const body = sections
    .filter(({ key }) => structured?.[key] !== null && structured?.[key] !== undefined)
    .map(
      ({ key, label }) => `
        <section class="reader-section">
          <h2>${escapeHtml(label)}</h2>
          ${renderStructuredValue(structured[key])}
        </section>
      `,
    )
    .join("");
  if (!body) return "";
  return `<div class="structured-reader"><h1>${escapeHtml(title)}</h1>${body}</div>`;
}

function renderStructuredValue(value) {
  if (value === null || value === undefined || value === "") {
    return "<p class='muted'>No data.</p>";
  }
  if (typeof value === "string") {
    return `<pre class="structured-text">${escapeHtml(value)}</pre>`;
  }
  if (typeof value !== "object") {
    return `<p>${escapeHtml(value)}</p>`;
  }
  if (Array.isArray(value)) {
    if (!value.length) return "<p class='muted'>No data.</p>";
    return `<div class="structured-list">${value.map(renderStructuredValue).join("")}</div>`;
  }
  if (
    Object.prototype.hasOwnProperty.call(value, "content") ||
    Object.prototype.hasOwnProperty.call(value, "message")
  ) {
    const metadata = Object.entries(value).filter(
      ([key]) => key !== "content" && key !== "message",
    );
    const metadataHtml = metadata.length
      ? `<dl class="reader-metadata">${metadata
          .map(
            ([key, entry]) => `
              <dt>${escapeHtml(titleFromKey(key))}</dt>
              <dd>${formatDetailValue(entry)}</dd>
            `,
          )
          .join("")}</dl>`
      : "";
    const content = value.content ?? value.message;
    return `
      ${metadataHtml}
      ${content ? `<pre class="structured-text">${escapeHtml(content)}</pre>` : ""}
    `;
  }
  return Object.entries(value)
    .map(
      ([key, entry]) => `
        <section class="reader-subsection">
          <h3>${escapeHtml(titleFromKey(key))}</h3>
          ${renderStructuredValue(entry)}
        </section>
      `,
    )
    .join("");
}

function openReportViewer(kind) {
  const payload = state.currentDrilldown;
  if (!payload) return;
  const structured = payload.structured || {};
  const isArticle = kind === "article";
  const isDecision = kind === "decision";
  const isDiagnostics = kind === "diagnostics";
  let html = isArticle ? payload.article_markdown_html : payload.report_markdown_html;
  if (isDecision) {
    html = renderStructuredDocument("Decision Detail", structured, [
      { key: "prediction_snapshot", label: "Prediction Summary" },
      { key: "final_decision", label: "Final Decision" },
      { key: "investment_analysis", label: "Investment Analysis" },
      { key: "risk_analysis", label: "Risk Analysis" },
    ]);
  } else if (isDiagnostics) {
    html = renderStructuredDocument("Agent Outputs", structured, [
      { key: "reports", label: "Reports" },
      { key: "artifact_statuses", label: "Artifact Statuses" },
      { key: "analysis_validity", label: "Analysis Validity" },
    ]);
  }
  if (!html) return;
  const titleKind = isArticle
    ? "Article"
    : isDecision
      ? "Decision Detail"
      : isDiagnostics
        ? "Agent Outputs"
        : "Report";
  const title = `${payload.ticker_ibkr || payload.ticker_yf || "Equity"} ${titleKind}`;
  elements.reportViewerTitle().textContent = title;
  elements.reportViewerBody().innerHTML = html;
  elements.reportViewer().classList.remove("hidden");
  elements.reportViewer().setAttribute("aria-hidden", "false");
}

function closeReportViewer() {
  const viewer = elements.reportViewer();
  viewer.classList.add("hidden");
  viewer.setAttribute("aria-hidden", "true");
  elements.reportViewerTitle().textContent = "Report";
  elements.reportViewerBody().innerHTML = "";
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
        {
          label: "Listing mapping",
          value:
            position.ticker_identity_verified === false
              ? `Review required (${position.ticker_resolution_source || "unresolved"})`
              : undefined,
        },
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
        {
          label: "Valuation status",
          value:
            position.valuation_valid === false
              ? position.valuation_issue || "Review required"
              : undefined,
        },
        {
          label: "Market value",
          value:
            position.valuation_valid === false
              ? "Unavailable"
              : fmtCurrency(position.market_value_usd),
        },
        {
          label: "Local-price return",
          value:
            position.local_return_pct !== undefined &&
            position.local_return_pct !== null
              ? `${Number(position.local_return_pct).toFixed(1)}%`
              : undefined,
        },
        {
          label: "Implied FX / basis",
          value:
            position.fx_effect_pct !== undefined &&
            position.fx_effect_pct !== null
              ? `${Number(position.fx_effect_pct).toFixed(1)}%`
              : undefined,
        },
        { label: "Return note", value: position.fx_return_issue },
        {
          label: "Unrealized P/L (USD)",
          value:
            position.valuation_valid === false
              ? "Unavailable"
              : fmtCurrency(position.unrealized_pnl_usd),
        },
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
      ${renderDetailSection("Thesis and Valuation", [
        {
          label: "Thesis break",
          value:
            Array.isArray(analysis.kill_criteria) && analysis.kill_criteria.length
              ? analysis.kill_criteria.join(" · ")
              : undefined,
        },
        {
          label: "Entry",
          value:
            analysis.entry_price !== undefined
              ? fmtLocalMoney(analysis.entry_price, analysis.currency)
              : undefined,
        },
        {
          label: "Downside valuation review",
          value:
            analysis.stop_price !== undefined
              ? fmtLocalMoney(analysis.stop_price, analysis.currency)
              : undefined,
        },
        {
          label: "Base-case ref",
          value:
            analysis.target_1_price !== undefined
              ? fmtLocalMoney(analysis.target_1_price, analysis.currency)
              : undefined,
        },
        {
          label: "Stretch ref",
          value:
            analysis.target_2_price !== undefined
              ? fmtLocalMoney(analysis.target_2_price, analysis.currency)
              : undefined,
        },
        { label: "Standalone verdict", value: tradeBlock.action },
        { label: "Target size %", value: tradeBlock.size_pct },
      ])}
    </div>
    ${renderReportActions(payload)}
  `;
}
