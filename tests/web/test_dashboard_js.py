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
  source + "\\nglobalThis.__dashboardTest = {{ escapeHtmlText, escapeHtmlAttr, renderTickerLink, renderSettings, renderConcentrationHeader, updateMacroAlert, state, elements }};",
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
    result = _run_dashboard_js(
        """
const { escapeHtmlText, escapeHtmlAttr } = __dashboardTest;
return {
  text: escapeHtmlText('A "quote" & <tag>'),
  attr: escapeHtmlAttr('A "quote" & <tag>'),
};
"""
    )

    assert result["text"] == 'A "quote" &amp; &lt;tag&gt;'
    assert result["attr"] == "A &quot;quote&quot; &amp; &lt;tag&gt;"


def test_render_ticker_link_escapes_data_attribute_quotes():
    html = _run_dashboard_js(
        """
const { renderTickerLink } = __dashboardTest;
return renderTickerLink({
  ticker_yf: 'BMW.DE" data-pwned="1',
  ticker_ibkr: "BMW",
});
"""
    )

    assert 'data-ticker="BMW.DE&quot; data-pwned=&quot;1"' in html
    assert 'data-ticker="BMW.DE" data-pwned="1"' not in html


def test_render_settings_escapes_input_value_attributes():
    html = _run_dashboard_js(
        """
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
"""
    )

    assert 'value="U123&quot; autofocus onfocus=&quot;alert(1)' in html
    assert 'value="U123" autofocus onfocus="alert(1)' not in html
    assert 'value="watchlist&quot; data-extra=&quot;1"' in html


def test_render_concentration_header_escapes_attribute_contexts():
    html = _run_dashboard_js(
        """
const { renderConcentrationHeader } = __dashboardTest;
return renderConcentrationHeader('sector" data-breakout="1', 'weight', 'Top "Weight"');
"""
    )

    assert 'data-sort-section="sector&quot; data-breakout=&quot;1"' in html
    assert (
        'aria-label="Sort sector&quot; data-breakout=&quot;1 concentration by top &quot;weight&quot;"'
        in html
    )
    assert 'title="Sort by Top &quot;Weight&quot;"' in html


def test_update_macro_alert_escapes_headline_markup():
    result = _run_dashboard_js(
        """
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
"""
    )

    assert "<img" not in result
    assert "&lt;img src=x onerror=alert(1)&gt;" in result
