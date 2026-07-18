from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.web.ibkr_dashboard.views import DASHBOARD_SCRIPTS

_DASHBOARD_STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "web" / "ibkr_dashboard" / "static"
)
_DASHBOARD_JS_FILES = tuple(
    _DASHBOARD_STATIC / filename for filename in DASHBOARD_SCRIPTS
)


def _run_dashboard_js(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for dashboard.js regression tests")

    script = f"""
const fs = require("fs");
const vm = require("vm");

const sourcePaths = {json.dumps([str(path) for path in _DASHBOARD_JS_FILES])};
let source = sourcePaths.map((path) => fs.readFileSync(path, "utf8")).join("\\n");
source = source.replace(/\\ninitializeDashboard\\(\\);\\s*$/, "\\n");

function makeElement() {{
  return {{
    classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
    addEventListener() {{}},
    setAttribute() {{}},
    textContent: "",
    innerHTML: "",
    value: "",
    dataset: {{}},
  }};
}}

const context = {{
  console,
  fetch: async () => ({{ ok: true, json: async () => ({{}}) }}),
  setInterval: () => 1,
  clearInterval: () => {{}},
  FormData: class {{
    get() {{
      return null;
    }}
  }},
  document: {{
    querySelectorAll: () => [],
    getElementById: () => makeElement(),
  }},
}};

    vm.createContext(context);
    vm.runInContext(
      source + "\\nglobalThis.__dashboardTest = {{ escapeHtmlText, escapeHtmlAttr, renderTickerLink, renderSettings, renderConcentrationHeader, renderConcentrationCard, renderActiveTab, renderRefresh, renderActions, renderWatchlist, renderOrders, renderDrilldown, openReportViewer, closeReportViewer, updateMacroAlert, updateModeAlert, updateReloadAlert, updateStatus, fmtLocalMoney, fmtScorePct, reasonHead, normalizeReason, renderActionTable, state, elements }};",
      context,
    );
const __dashboardTest = context.__dashboardTest;

const result = (() => {{
{expression}
}})();

console.log(JSON.stringify(result));
"""
    completed = subprocess.run(
        # Use an absolute executable path and keep file descriptors open so
        # CPython can take the safer posix_spawn path on macOS/Python 3.12.
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        close_fds=False,
    )
    return json.loads(completed.stdout)


def test_escape_html_attr_encodes_quotes_without_overescaping_text_context():
    result = _run_dashboard_js("""
const { escapeHtmlText, escapeHtmlAttr } = __dashboardTest;
return {
  text: escapeHtmlText('A "quote" & <tag>'),
  attr: escapeHtmlAttr('A "quote" & <tag>'),
};
""")

    assert result["text"] == 'A "quote" &amp; &lt;tag&gt;'
    assert result["attr"] == "A &quot;quote&quot; &amp; &lt;tag&gt;"


def test_render_ticker_link_escapes_data_attribute_quotes():
    html = _run_dashboard_js("""
const { renderTickerLink } = __dashboardTest;
return renderTickerLink({
  ticker_yf: 'BMW.DE" data-pwned="1',
  ticker_ibkr: "BMW",
});
""")

    assert 'data-ticker="BMW.DE&quot; data-pwned=&quot;1"' in html
    assert 'data-ticker="BMW.DE" data-pwned="1"' not in html


def test_render_settings_escapes_input_value_attributes():
    html = _run_dashboard_js("""
const { renderSettings, state } = __dashboardTest;
state.settings = {
  account_id: 'U123" autofocus onfocus="alert(1)',
  watchlist_name: 'watchlist" data-extra="1',
  read_only: false,
  max_age_days: 14,
  refresh_limit: 10,
  quick_mode_default: true,
  notes: 'notes " stay in textarea text',
};
return renderSettings();
""")

    assert 'value="U123&quot; autofocus onfocus=&quot;alert(1)' in html
    assert 'value="U123" autofocus onfocus="alert(1)' not in html
    assert 'value="watchlist&quot; data-extra=&quot;1"' in html


def test_render_settings_uses_loaded_values_not_fallback_literals():
    html = _run_dashboard_js("""
const { renderSettings, state } = __dashboardTest;
state.settings = {
  read_only: false,
  max_age_days: 37,
  refresh_limit: 4,
  quick_mode_default: false,
};
return renderSettings();
""")

    assert '<input name="max_age_days" type="number" value="37">' in html
    assert '<input name="refresh_limit" type="number" value="4">' in html
    assert '<input name="max_age_days" type="number" value="14">' not in html
    assert '<input name="refresh_limit" type="number" value="10">' not in html


def test_render_settings_falls_back_only_when_settings_values_absent():
    html = _run_dashboard_js("""
const { renderSettings, state } = __dashboardTest;
state.settings = {
  read_only: true,
};
return renderSettings();
""")

    assert '<input name="max_age_days" type="number" value="14">' in html
    assert '<input name="refresh_limit" type="number" value="10">' in html


def test_render_refresh_uses_portfolio_manager_freshness_wording():
    html = _run_dashboard_js("""
const { renderRefresh, state } = __dashboardTest;
state.snapshot = {
  freshness: {
    blocking_now: [{ ticker: "7203.T" }],
    stale_in_queue: [{ ticker: "0005.HK" }],
    candidate_blocked: [{ ticker: "3515.TW" }],
    due_soon: [{ ticker: "6831.HK" }],
    fresh_count: 2,
  },
  screening_freshness: {
    status: "fresh",
    screening_date: "2026-07-15",
    age_days: 0,
    candidate_count: 10,
    buy_count: 2,
  },
};
return renderRefresh();
""")

    assert "Needs review" in html
    assert "Refresh queue" in html
    assert "Needs full refresh" in html
    assert "Queue action-required reruns (2)" in html
    assert "Blocking now" not in html
    assert "Stale in queue" not in html


def test_loading_copy_explains_live_ibkr_fetch():
    result = _run_dashboard_js("""
const { renderActiveTab, updateStatus, state, elements } = __dashboardTest;
const tab = { innerHTML: "" };
const status = { textContent: "" };
const context = { textContent: "" };
elements.tabContent = () => tab;
elements.status = () => status;
elements.context = () => context;
state.settings = { read_only: false };
state.snapshot = null;
state.snapshotMeta = {
  status: "loading",
  fetched_at: null,
  cache_hit: false,
  refreshing: true,
  last_error: null,
};
renderActiveTab();
updateStatus();
return {
  tab: tab.innerHTML,
  status: status.textContent,
  context: context.textContent,
};
""")

    expected = "Fetching IBKR positions, watchlist, and orders; may take a few minutes."
    assert expected in result["tab"]
    assert result["status"] == "Loading live data…"
    assert result["context"] == expected


def test_loading_copy_stays_generic_in_read_only_mode():
    result = _run_dashboard_js("""
const { renderActiveTab, updateStatus, state, elements } = __dashboardTest;
const tab = { innerHTML: "" };
const status = { textContent: "" };
const context = { textContent: "" };
elements.tabContent = () => tab;
elements.status = () => status;
elements.context = () => context;
state.settings = { read_only: true };
state.snapshot = null;
state.snapshotMeta = {
  status: "loading",
  fetched_at: null,
  cache_hit: false,
  refreshing: true,
  last_error: null,
};
renderActiveTab();
updateStatus();
return {
  tab: tab.innerHTML,
  status: status.textContent,
  context: context.textContent,
};
""")

    assert "Loading current data…" in result["tab"]
    assert result["status"] == "Loading data…"
    assert result["context"] == "Loading current data…"


def test_render_watchlist_zero_cost_buy_uses_short_na_label():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: {
    name: "watchlist-2026",
    total: 2,
    tickers: ["3393.T", "3762.T"],
  },
  actions: {
    watchlist_buy: [
      {
        ticker_yf: "3393.T",
        ticker_ibkr: "3393",
        action: "BUY",
        reason: "Watchlist BUY (2026-07-05) — Medium conviction, target 4.0%",
        suggested_price: 2980,
        suggested_quantity: null,
        cash_impact_usd: 0,
      },
      {
        ticker_yf: "3762.T",
        ticker_ibkr: "3762",
        action: "BUY",
        reason: "Watchlist BUY (2026-07-11) — Medium conviction, target 4.0%",
        suggested_price: 1800,
        suggested_quantity: 100,
        cash_impact_usd: -1206,
      },
    ],
    watchlist_candidate: [],
    watchlist_monitor: [],
    watchlist_remove: [],
  },
};
return renderWatchlist();
""")

    assert ">N/A</td>" in html
    assert "$1,206" in html
    assert "$0" not in html


def test_render_watchlist_withheld_table_shows_breach():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "watchlist-2026", total: 2, tickers: ["7203.T"] },
  actions: {
    watchlist_buy: [],
    watchlist_candidate: [],
    watchlist_monitor: [],
    watchlist_remove: [
      {
        ticker_yf: "7203.T",
        ticker_ibkr: "7203",
        action: "BUY",
        reason: "Watchlist BUY — Medium conviction",
        removal_reason: "concentration_displaced",
        concentration: "overweight exchange T (projected 49.0% > 40%)",
      },
    ],
    watchlist_withheld: [
      {
        ticker_yf: "9984.T",
        ticker_ibkr: "9984",
        action: "BUY",
        reason: "New BUY — Medium conviction",
        concentration: "overweight exchange T (projected 49.0% > 40%)",
      },
    ],
  },
};
return renderWatchlist();
""")

    assert "Withheld By Concentration" in html
    assert "9984" in html
    assert html.count("overweight exchange T (projected 49.0% &gt; 40%)") == 2


def test_render_watchlist_withheld_table_absent_when_missing_or_empty():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "watchlist-2026", total: 1, tickers: ["7203.T"] },
  actions: {
    watchlist_buy: [],
    watchlist_candidate: [],
    watchlist_monitor: [],
    watchlist_remove: [],
  },
};
return renderWatchlist();
""")

    assert "Withheld By Concentration" not in html


def test_render_watchlist_withheld_entry_without_concentration_falls_back():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "watchlist-2026", total: 1, tickers: [] },
  actions: {
    watchlist_buy: [],
    watchlist_candidate: [],
    watchlist_monitor: [],
    watchlist_remove: [],
    watchlist_withheld: [
      {
        ticker_yf: "9984.T",
        ticker_ibkr: "9984",
        action: "BUY",
        reason: "New BUY — Medium conviction",
      },
    ],
  },
};
return renderWatchlist();
""")

    assert "Withheld By Concentration" in html
    assert "New BUY — Medium conviction" in html


def test_render_watchlist_distinguishes_unavailable_from_empty():
    unavailable = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "missing", total: null, tickers: [], status: "unavailable" },
  actions: { watchlist_buy: [], watchlist_candidate: [], watchlist_monitor: [], watchlist_remove: [] },
};
return renderWatchlist();
""")
    empty = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "empty", total: 0, tickers: [], status: "loaded" },
  actions: { watchlist_buy: [], watchlist_candidate: [], watchlist_monitor: [], watchlist_remove: [] },
};
return renderWatchlist();
""")

    assert "IBKR Watchlist Unavailable" in unavailable
    assert "Membership is unknown" in unavailable
    assert "Loaded IBKR Watchlist: empty" in empty
    assert "No tickers were loaded" in empty


def test_render_watchlist_labels_in_flight_membership_unknown():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: null, total: null, tickers: [], status: "not_loaded" },
  actions: {
    watchlist_buy: [], watchlist_candidate: [], watchlist_monitor: [], watchlist_remove: [],
    watchlist_in_flight: [{
      ticker_yf: "WDO.TO", ticker_ibkr: "WDO", action: "BUY",
      reason: "New BUY", watchlist_membership: "unknown",
    }],
  },
};
return renderWatchlist();
""")

    assert "No IBKR Watchlist Loaded" in html
    assert "BUY Orders Already In Flight" in html
    assert "Membership unknown" in html


def test_render_drilldown_keeps_long_structured_content_out_of_side_panel():
    html = _run_dashboard_js("""
const { renderDrilldown } = __dashboardTest;
return renderDrilldown({
  ticker_ibkr: "3393",
  action: "BUY",
  reason: "Watchlist BUY",
  urgency: "MEDIUM",
  position: {},
  analysis: {
    verdict: "BUY",
    analysis_date: "2026-07-05",
    health_adj: 92,
    growth_adj: 67,
    zone: "LOW",
    conviction: "Medium",
    trade_block: { action: "BUY", size_pct: 4.0, risk_reward: "3:1" },
  },
  structured: {
    final_decision: { block: "PM_BLOCK SHOULD NOT APPEAR" },
    investment_analysis: { thesis: "Long investment analysis should not appear" },
    risk_analysis: { risk: "Long risk analysis should not appear" },
    artifact_statuses: { report: "valid" },
    analysis_validity: { auditor_report: { content: "Raw diagnostic payload" } },
  },
  report_markdown_html: "<h1>Full Report Body</h1><p>Readable report</p>",
  article_markdown_html: null,
});
""")

    assert "Open Report" in html
    assert "Decision Detail" in html
    assert "Agent Outputs" in html
    assert "Artifact Statuses" not in html
    assert "Analysis Validity" not in html
    assert "Raw diagnostic payload" not in html
    assert "PM_BLOCK SHOULD NOT APPEAR" not in html
    assert "Long investment analysis should not appear" not in html
    assert "Long risk analysis should not appear" not in html
    assert "Full Report Body" not in html


def test_open_report_viewer_renders_selected_report_html():
    result = _run_dashboard_js("""
const { openReportViewer, state, elements } = __dashboardTest;
const viewer = {
  classList: { add() {}, remove() {} },
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
};
const title = { textContent: "" };
const body = { innerHTML: "" };
elements.reportViewer = () => viewer;
elements.reportViewerTitle = () => title;
elements.reportViewerBody = () => body;
state.currentDrilldown = {
  ticker_ibkr: "3393",
  report_markdown_html: "<h1>Full Report</h1><p>Decision text</p>",
  article_markdown_html: "<h1>Article</h1>",
};
openReportViewer("report");
return {
  title: title.textContent,
  body: body.innerHTML,
  hidden: viewer.attributes["aria-hidden"],
};
""")

    assert result["title"] == "3393 Report"
    assert "<h1>Full Report</h1>" in result["body"]
    assert "Decision text" in result["body"]
    assert result["hidden"] == "false"


def test_open_decision_detail_viewer_renders_structured_content_on_demand():
    result = _run_dashboard_js("""
const { openReportViewer, state, elements } = __dashboardTest;
const viewer = {
  classList: { add() {}, remove() {} },
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
};
const title = { textContent: "" };
const body = { innerHTML: "" };
elements.reportViewer = () => viewer;
elements.reportViewerTitle = () => title;
elements.reportViewerBody = () => body;
state.currentDrilldown = {
  ticker_ibkr: "3393",
  structured: {
    final_decision: {
      verdict: "BUY",
      rationale: "Readable decision detail",
    },
  },
};
openReportViewer("decision");
return {
  title: title.textContent,
  body: body.innerHTML,
  hidden: viewer.attributes["aria-hidden"],
};
""")

    assert result["title"] == "3393 Decision Detail"
    assert "Final Decision" in result["body"]
    assert "Readable decision detail" in result["body"]
    assert result["hidden"] == "false"


def test_open_agent_outputs_viewer_escapes_structured_content():
    result = _run_dashboard_js("""
const { openReportViewer, state, elements } = __dashboardTest;
const viewer = {
  classList: { add() {}, remove() {} },
  attributes: {},
  setAttribute(name, value) { this.attributes[name] = value; },
};
const title = { textContent: "" };
const body = { innerHTML: "" };
elements.reportViewer = () => viewer;
elements.reportViewerTitle = () => title;
elements.reportViewerBody = () => body;
state.currentDrilldown = {
  ticker_ibkr: "3393",
  structured: {
    artifact_statuses: {
      auditor_report: {
        complete: true,
        ok: true,
        content: "Auditor text <img src=x onerror=alert(1)>",
      },
    },
  },
};
openReportViewer("diagnostics");
return {
  title: title.textContent,
  body: body.innerHTML,
  hidden: viewer.attributes["aria-hidden"],
};
""")

    assert result["title"] == "3393 Agent Outputs"
    assert "Auditor Report" in result["body"]
    assert "&lt;img src=x onerror=alert(1)&gt;" in result["body"]
    assert "<img" not in result["body"]
    assert result["hidden"] == "false"


def test_render_concentration_header_escapes_attribute_contexts():
    html = _run_dashboard_js("""
const { renderConcentrationHeader } = __dashboardTest;
return renderConcentrationHeader('sector" data-breakout="1', 'weight', 'Top "Weight"');
""")

    assert 'data-sort-section="sector&quot; data-breakout=&quot;1"' in html
    assert (
        'aria-label="Sort sector&quot; data-breakout=&quot;1 concentration by top &quot;weight&quot;"'
        in html
    )
    assert 'title="Sort by Top &quot;Weight&quot;"' in html


def test_update_macro_alert_escapes_headline_markup():
    result = _run_dashboard_js("""
const { updateMacroAlert, state, elements } = __dashboardTest;
const alert = {
  classList: { add() {}, remove() {} },
  textContent: "",
  innerHTML: "",
};
elements.macroAlert = () => alert;
state.snapshot = {
  macro_alert: {
    detected: true,
    headline: 'Shock <img src=x onerror=alert(1)>',
    correlation_pct: 75,
  },
};
updateMacroAlert();
return alert.innerHTML;
""")

    assert "<img" not in result
    assert "&lt;img src=x onerror=alert(1)&gt;" in result


def test_update_mode_alert_uses_configured_results_dir():
    result = _run_dashboard_js("""
const { updateModeAlert, state, elements } = __dashboardTest;
const alert = {
  classList: { add() {}, remove() {} },
  textContent: "",
  innerHTML: "",
};
elements.modeAlert = () => alert;
state.snapshot = { read_only: true };
state.settings = {
  results_dir: 'scratch/results-custom',
  account_id: null,
};
updateModeAlert();
return alert.innerHTML;
""")

    assert "scratch/results-custom" in result
    assert "<code>results/</code>" not in result


def test_fmt_local_money_uses_thousands_separators():
    result = _run_dashboard_js("""
const { fmtLocalMoney } = __dashboardTest;
return {
  jpy: fmtLocalMoney(24500, "jpy"),
  none: fmtLocalMoney(null, "JPY"),
  no_currency: fmtLocalMoney(12.5, null),
  non_numeric: fmtLocalMoney("abc", "USD"),
};
""")

    assert result["jpy"] == "JPY 24,500.00"
    assert result["none"] == "—"
    assert result["no_currency"] == "? 12.50"
    assert result["non_numeric"] == "—"


def test_fmt_score_pct_renders_whole_percent():
    result = _run_dashboard_js("""
const { fmtScorePct } = __dashboardTest;
return {
  score: fmtScorePct(72),
  float: fmtScorePct(66.7),
  missing: fmtScorePct(null),
  blank: fmtScorePct(""),
};
""")

    assert result["score"] == "72%"
    assert result["float"] == "67%"
    assert result["missing"] == "—"
    assert result["blank"] == "—"


def test_reason_head_splits_at_dash_and_normalizes_tokens():
    result = _run_dashboard_js("""
const { reasonHead } = __dashboardTest;
return {
  split: reasonHead("Watchlist BUY (2026-07-05) — Medium conviction, target 4.0%"),
  no_dash: reasonHead("Position remains within thesis"),
  empty: reasonHead(""),
  verdict: reasonHead("Verdict → DO_NOT_INITIATE — stale analysis"),
  dash_no_detail: reasonHead("Something — "),
};
""")

    assert result["split"] == "Watchlist BUY (2026-07-05)"
    assert result["no_dash"] == "Position remains within thesis"
    assert result["empty"] == "—"
    # DO_NOT_INITIATE → REJECT and Verdict → : normalization applied before split.
    assert result["verdict"] == "Verdict: REJECT"
    assert result["dash_no_detail"] == "Something — "


def test_render_action_table_puts_reason_head_in_cell_and_full_in_tooltip():
    html = _run_dashboard_js("""
const { renderActionTable } = __dashboardTest;
return renderActionTable("Sells", [
  {
    ticker_yf: "7203.T",
    ticker_ibkr: "7203",
    action: "SELL",
    reason: "Confirmed thesis failure — health 42% below gate, two full analyses agree",
  },
], []);
""")

    assert "Confirmed thesis failure</td>" in html
    assert (
        'title="Confirmed thesis failure — health 42% below gate, two full analyses agree"'
        in html
    )


def test_render_action_table_omit_reason_drops_reason_column():
    html = _run_dashboard_js("""
const { renderActionTable } = __dashboardTest;
return renderActionTable("Holds", [
  { ticker_yf: "7203.T", ticker_ibkr: "7203", action: "HOLD", reason: "Position remains within thesis" },
], [{ label: "Weight", numeric: true, render: () => "5.0%" }], { omitReason: true });
""")

    assert "<th>Reason</th>" not in html
    assert "Position remains within thesis" not in html
    assert '<th class="num">Weight</th>' in html
    assert '<td class="num">5.0%</td>' in html


def test_render_hold_row_shows_weight_gain_stop_target():
    html = _run_dashboard_js("""
const { renderActions, state } = __dashboardTest;
state.snapshot = {
  portfolio: { net_liquidation_usd: 100000 },
  actions: {
    action_sections: [
      {
        key: "hold",
        title: "Holds",
        kind: "reconciliation_items",
        items: [
          {
            ticker_yf: "7203.T",
            ticker_ibkr: "7203",
            action: "HOLD",
            reason: "Position remains within thesis",
            analysis: { entry_price: 2000, stop_price: 1700, target_1_price: 2600, currency: "JPY" },
            position: { current_price_local: 2300, currency: "JPY", market_value_usd: 15000 },
          },
        ],
      },
    ],
  },
};
return renderActions();
""")

    assert "<th>Reason</th>" not in html
    assert "15.0%" in html  # weight = 15000 / 100000
    assert "+15.0%" in html  # gain = (2300 - 2000) / 2000
    assert "JPY 1,700.00" in html  # stop
    assert "JPY 2,600.00" in html  # target


def test_render_hold_row_falls_back_to_cost_basis_for_entry():
    html = _run_dashboard_js("""
const { renderActions, state } = __dashboardTest;
state.snapshot = {
  portfolio: { net_liquidation_usd: 100000 },
  actions: {
    action_sections: [
      {
        key: "hold",
        title: "Holds",
        kind: "reconciliation_items",
        items: [
          {
            ticker_yf: "MEGP.L",
            ticker_ibkr: "MEGP",
            action: "HOLD",
            reason: "Position remains within thesis",
            analysis: null,
            position: { avg_cost_local: 100, current_price_local: 120, currency: "GBP", market_value_usd: 5000 },
          },
        ],
      },
    ],
  },
};
return renderActions();
""")

    assert "+20.0%" in html  # gain from cost basis 100 → 120
    assert 'title="Cost basis (average cost) — no analysis entry price"' in html


def test_render_concentration_card_shows_limit_and_warns_near_cap():
    html = _run_dashboard_js("""
const { renderConcentrationCard } = __dashboardTest;
return renderConcentrationCard(
  "exchange",
  "Exchange Concentration",
  "Exchange",
  { T: 38, US: 20 },
  "No positions.",
  40,
);
""")

    assert "limit 40%" in html
    # T at 38 is >= 90% of 40 (=36) → warn; US at 20 is not.
    assert 'class="conc-warn"' in html
    assert "⚠" in html


def test_render_concentration_card_without_limit_has_no_warn():
    html = _run_dashboard_js("""
const { renderConcentrationCard } = __dashboardTest;
return renderConcentrationCard(
  "exchange",
  "Exchange Concentration",
  "Exchange",
  { T: 38 },
  "No positions.",
  null,
);
""")

    assert "limit" not in html
    assert "conc-warn" not in html


def test_render_watchlist_remove_uses_structured_breaches():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "wl", total: 1, tickers: ["7203.T"] },
  actions: {
    watchlist_buy: [], watchlist_candidate: [], watchlist_monitor: [],
    watchlist_remove: [
      {
        ticker_yf: "7203.T", ticker_ibkr: "7203", action: "BUY",
        reason: "Watchlist BUY", removal_reason: "concentration_displaced",
        concentration: "overweight exchange T (projected 52.7% > 40%)",
        breaches: [{ dimension: "exchange", key: "T", projected_pct: 52.7, limit_pct: 40 }],
      },
    ],
  },
};
return renderWatchlist();
""")

    assert "exchange T 52.7% &gt; 40%" in html


def test_render_withheld_groups_same_category_into_one_row():
    html = _run_dashboard_js("""
const { renderWatchlist, state } = __dashboardTest;
state.snapshot = {
  watchlist: { name: "wl", total: 1, tickers: [] },
  actions: {
    watchlist_buy: [], watchlist_candidate: [], watchlist_monitor: [], watchlist_remove: [],
    watchlist_withheld: [
      {
        ticker_yf: "9984.T", ticker_ibkr: "9984", action: "BUY", reason: "New BUY",
        concentration: "overweight exchange T (projected 49.0% > 40%)",
        breaches: [{ dimension: "exchange", key: "T", projected_pct: 49.0, limit_pct: 40 }],
      },
      {
        ticker_yf: "6758.T", ticker_ibkr: "6758", action: "BUY", reason: "New BUY",
        concentration: "overweight exchange T (projected 52.7% > 40%)",
        breaches: [{ dimension: "exchange", key: "T", projected_pct: 52.7, limit_pct: 40 }],
      },
    ],
  },
};
return renderWatchlist();
""")

    # Two names in the same overweight bucket collapse to a single grouped row.
    assert "Withheld By Concentration" in html
    assert "exchange T up to 52.7% &gt; 40%" in html
    assert "9984, 6758" in html or "6758, 9984" in html
    assert html.count("<tbody>") == 1


def test_render_orders_shows_failure_banner_distinct_from_empty():
    failed = _run_dashboard_js("""
const { renderOrders, state } = __dashboardTest;
state.snapshot = {
  read_only: false,
  orders: [],
  cash_summary: {},
  errors: { live_orders: "IBKR session not authenticated" },
};
return renderOrders();
""")
    empty = _run_dashboard_js("""
const { renderOrders, state } = __dashboardTest;
state.snapshot = {
  read_only: false,
  orders: [],
  cash_summary: {},
  errors: {},
};
return renderOrders();
""")

    assert "Live orders unavailable" in failed
    assert "IBKR session not authenticated" in failed
    assert "Live-order data could not be loaded." in failed
    assert "Live orders unavailable" not in empty
    assert "No live orders." in empty


def test_update_reload_alert_prompts_when_job_completes_after_cached_snapshot():
    shown = _run_dashboard_js("""
const { updateReloadAlert, state, elements } = __dashboardTest;
const alert = { classList: { add() {}, remove() {} }, innerHTML: "" };
elements.reloadAlert = () => alert;
state.snapshot = { cache_hit: true, as_of: "2026-07-18T10:00:00+00:00" };
state.jobs = [{ status: "completed", finished_at: "2026-07-18T11:00:00+00:00" }];
state.reloadDismissedAt = null;
updateReloadAlert();
return alert.innerHTML;
""")
    stale_job = _run_dashboard_js("""
const { updateReloadAlert, state, elements } = __dashboardTest;
const alert = { classList: { add() {}, remove() {} }, innerHTML: "" };
elements.reloadAlert = () => alert;
state.snapshot = { cache_hit: true, as_of: "2026-07-18T12:00:00+00:00" };
state.jobs = [{ status: "completed", finished_at: "2026-07-18T11:00:00+00:00" }];
state.reloadDismissedAt = null;
updateReloadAlert();
return alert.innerHTML;
""")
    fresh_snapshot = _run_dashboard_js("""
const { updateReloadAlert, state, elements } = __dashboardTest;
const alert = { classList: { add() {}, remove() {} }, innerHTML: "" };
elements.reloadAlert = () => alert;
state.snapshot = { cache_hit: false, as_of: "2026-07-18T10:00:00+00:00" };
state.jobs = [{ status: "completed", finished_at: "2026-07-18T11:00:00+00:00" }];
state.reloadDismissedAt = null;
updateReloadAlert();
return alert.innerHTML;
""")

    assert "Analyses refreshed since this data was loaded." in shown
    assert stale_job == ""
    assert fresh_snapshot == ""
