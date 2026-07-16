from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_DASHBOARD_JS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "web"
    / "ibkr_dashboard"
    / "static"
    / "dashboard.js"
)


def _run_dashboard_js(expression: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for dashboard.js regression tests")

    script = f"""
const fs = require("fs");
const vm = require("vm");

const sourcePath = {json.dumps(str(_DASHBOARD_JS))};
let source = fs.readFileSync(sourcePath, "utf8");
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
      source + "\\nglobalThis.__dashboardTest = {{ escapeHtmlText, escapeHtmlAttr, renderTickerLink, renderSettings, renderConcentrationHeader, renderActiveTab, renderRefresh, renderWatchlist, renderDrilldown, openReportViewer, closeReportViewer, updateMacroAlert, updateModeAlert, updateStatus, state, elements }};",
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
