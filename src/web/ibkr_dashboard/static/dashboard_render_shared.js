function fmtCurrency(value) {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function fmtLocalMoney(value, currency) {
  if (value === null || value === undefined) return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const num = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(number);
  if (!currency) return `? ${num}`;
  return `${currency.toUpperCase()} ${num}`;
}

// Health / growth scores arrive on a 0–100 scale; render them as whole percents
// so a column of "72%" reads consistently instead of raw floats like "72.0".
function fmtScorePct(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return escapeHtml(String(value));
  return `${number.toFixed(0)}%`;
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

function fmtBuyCost(item) {
  if (item.cash_impact_usd === null || item.cash_impact_usd === undefined) {
    return "N/A";
  }
  const cost = Math.abs(Number(item.cash_impact_usd));
  if (!Number.isFinite(cost) || cost <= 0) {
    return "N/A";
  }
  return fmtCurrency(cost);
}

function escapeHtmlText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeHtmlAttr(value) {
  return escapeHtmlText(value)
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

const escapeHtml = escapeHtmlText;

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
  return `<button type="button" class="ticker-link" data-ticker="${escapeHtmlAttr(ticker)}">${escapeHtml(item.ticker_ibkr || ticker)}</button>`;
}

// Mirror of the CLI's normalize_reason (src/ibkr/portfolio_report_formatting.py):
// keep the two substitutions in lockstep so dashboard heads match the report.
function normalizeReason(reason) {
  return String(reason ?? "")
    .replaceAll("DO_NOT_INITIATE", "REJECT")
    .replaceAll("Verdict → ", "Verdict: ");
}

// Mirror of split_reason(): the head before " — "; the full text goes in a
// tooltip. Only split when there is real detail after the dash (else the whole
// reason is the head), matching the Python semantics exactly.
function reasonHead(reason) {
  const norm = normalizeReason(reason);
  if (!norm) return "—";
  const idx = norm.indexOf(" — ");
  if (idx > 0 && norm.slice(idx + 3).trim()) {
    return norm.slice(0, idx).trimEnd();
  }
  return norm;
}

// Columns are a uniform list of {label, render, numeric?, title?} so headers and
// cells share alignment classes and optional per-cell tooltips. The Ticker,
// Action and (unless omitted) Reason columns are always present; callers pass
// the trailing extras. omitReason drops the boilerplate Reason column (Holds).
function renderActionTable(title, items, extraColumns = [], options = {}) {
  if (!items || !items.length) {
    return `<section><h3 class="section-title">${escapeHtml(title)}</h3><p class="muted">None.</p></section>`;
  }
  const columns = [
    { label: "Ticker", render: (item) => renderTickerLink(item) },
    { label: "Action", render: (item) => escapeHtml(item.action) },
  ];
  if (!options.omitReason) {
    columns.push({
      label: "Reason",
      render: (item) => escapeHtml(reasonHead(item.reason)),
      title: (item) => normalizeReason(item.reason),
    });
  }
  columns.push(...extraColumns);
  const headers = columns
    .map((col) => `<th${col.numeric ? ' class="num"' : ""}>${escapeHtml(col.label)}</th>`)
    .join("");
  const rows = items
    .map((item) => {
      const cells = columns
        .map((col) => {
          const cls = col.numeric ? ' class="num"' : "";
          const title = col.title ? ` title="${escapeHtmlAttr(col.title(item))}"` : "";
          return `<td${cls}${title}>${col.render(item)}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `
    <section>
      <h3 class="section-title">${escapeHtml(title)}</h3>
      <table>
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>
  `;
}

function sellTypeLabel(item) {
  return item.sell_type_label || item.sell_type || "Sell";
}

function profitTakeDetail(item) {
  const parts = [];
  if (typeof item.cost_basis_return_pct === "number") {
    parts.push(`${item.cost_basis_return_pct.toFixed(1)}% gain`);
  }
  if (item.position?.tax_term) {
    parts.push(`tax: ${item.position.tax_term}`);
  }
  // item.reason is already the table's Reason column — do not repeat it here.
  return escapeHtml(parts.join(" · ") || "—");
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
          <td title="${escapeHtmlAttr(normalizeReason(item.reason))}">${escapeHtml(reasonHead(item.reason))}</td>
          <td class="num">${fmtScorePct(item.analysis?.health_adj)}</td>
          <td class="num">${fmtScorePct(item.analysis?.growth_adj)}</td>
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
        <thead><tr><th>Ticker</th><th>Reason</th><th class="num">Health</th><th class="num">Growth</th></tr></thead>
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

function getConcentrationEntries(weights) {
  return Object.entries(weights || {}).map(([label, weight]) => ({
    label,
    weight: Number(weight) || 0,
  }));
}

function getDefaultConcentrationDirection(key) {
  return key === "label" ? "asc" : "desc";
}

function getSortedConcentrationEntries(section, weights) {
  const sort = state.concentrationSorts[section] || {
    key: "weight",
    direction: "desc",
  };
  const multiplier = sort.direction === "asc" ? 1 : -1;
  return getConcentrationEntries(weights).sort((left, right) => {
    if (sort.key === "label") {
      const comparison = left.label.localeCompare(right.label, undefined, {
        sensitivity: "base",
      });
      if (comparison !== 0) {
        return comparison * multiplier;
      }
      return right.weight - left.weight;
    }

    const delta = left.weight - right.weight;
    if (delta !== 0) {
      return delta * multiplier;
    }
    return left.label.localeCompare(right.label, undefined, {
      sensitivity: "base",
    });
  });
}

function getSortArrow(section, key) {
  const sort = state.concentrationSorts[section];
  if (!sort || sort.key !== key) {
    return "↕";
  }
  return sort.direction === "asc" ? "↑" : "↓";
}

function renderConcentrationHeader(section, key, label) {
  const sort = state.concentrationSorts[section];
  const isActive = sort?.key === key;
  const ariaSort = isActive
    ? sort.direction === "asc"
      ? "ascending"
      : "descending"
    : "none";
  const classes = ["sort-button"];
  if (isActive) {
    classes.push("active");
  }
  return `
    <th aria-sort="${ariaSort}">
      <button
        type="button"
        class="${classes.join(" ")}"
        data-sort-section="${escapeHtmlAttr(section)}"
        data-sort-key="${escapeHtmlAttr(key)}"
        aria-label="${escapeHtmlAttr(`Sort ${section} concentration by ${label.toLowerCase()}`)}"
        title="${escapeHtmlAttr(`Sort by ${label}`)}"
      >
        <span>${escapeHtml(label)}</span>
        <span class="sort-indicator" aria-hidden="true">${getSortArrow(section, key)}</span>
      </button>
    </th>
  `;
}

function renderConcentrationCard(
  section,
  title,
  label,
  weights,
  emptyMessage,
  limitPct = null,
) {
  const hasLimit = typeof limitPct === "number" && Number.isFinite(limitPct);
  const rows = getSortedConcentrationEntries(section, weights)
    .map((entry) => {
      // Mirror the CLI convention: warn (amber ⚠) once a bucket reaches 90% of
      // its limit, so the raw percentage becomes actionable against the cap.
      const warn = hasLimit && entry.weight >= limitPct * 0.9;
      const pct = warn ? `${fmtPct(entry.weight)} ⚠` : fmtPct(entry.weight);
      return `
        <tr>
          <td>${escapeHtml(entry.label)}</td>
          <td class="${warn ? "conc-warn" : ""}">${pct}</td>
        </tr>
      `;
    })
    .join("");
  const body =
    rows ||
    `<tr><td colspan="2" class="muted">${escapeHtml(emptyMessage)}</td></tr>`;
  const limitNote = hasLimit
    ? ` <span class="muted conc-limit">limit ${Number(limitPct).toFixed(0)}%</span>`
    : "";
  return `
    <article class="card">
      <h3>${escapeHtml(title)}${limitNote}</h3>
      <table class="concentration-table">
        <thead>
          <tr>
            ${renderConcentrationHeader(section, "label", label)}
            ${renderConcentrationHeader(section, "weight", "%")}
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </article>
  `;
}
