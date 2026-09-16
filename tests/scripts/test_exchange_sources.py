"""Source completeness and Romanian screening boundary regressions."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import find_gems  # noqa: E402


@pytest.fixture
def bvb():
    config = json.loads(Path("config/exchanges.json").read_text())
    ex = next(ex for ex in config["exchanges"] if ex["yahoo_suffix"] == ".RO")
    return {**ex, "enabled": True, "min_expected_rows": 2}


def _html(symbols=("TLV", "H2O")):
    rows = "".join(
        f'<tr><td><a href="/FinancialInstruments/Details/FinancialInstrumentsDetails.aspx?s={s}">{s}</a> RO0000000000</td>'
        f"<td>{s} Société</td><td>34</td><td>0</td><td>15.09.2026</td><td>Premium</td></tr>"
        for s in symbols
    )
    return (
        '<input type="submit" id="ms1" disabled="disabled" value="Piata Reglementata">'
        '<button id="ms3">AeRO</button><table id="gv"><thead><tr>'
        + "".join(
            f"<th>{h}</th>"
            for h in (
                "Simbol / ISIN",
                "Societate",
                "Pret (RON)",
                "Var. (%)",
                "Data",
                "Categoria",
            )
        )
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _response(html):
    response = requests.Response()
    response.status_code = 200
    response._content = html.encode("utf-8")
    response.headers["Content-Type"] = "text/html"
    response.encoding = "ISO-8859-1"
    return response


def test_bvb_parser_keeps_symbol_separate_from_isin(bvb):
    session = MagicMock()
    session.get.return_value = _response(_html())
    df = find_gems._handle_bvb_shares(bvb, session)
    assert df.Symbol.tolist() == ["TLV", "H2O"]
    assert df.Company.tolist() == ["TLV Société", "H2O Société"]
    assert set(df.Currency) == {"RON"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.replace('disabled="disabled"', ""),
        lambda s: s.replace("Piata Reglementata", "AeRO"),
        lambda s: s.replace("Simbol / ISIN", "Symbol"),
        lambda s: s.replace("?s=TLV", "?s=OTHER"),
        lambda s: s.replace("<td>34</td>", ""),
    ],
)
def test_bvb_rejects_ambiguous_segment_or_identity(bvb, mutation):
    session = MagicMock()
    session.get.return_value = _response(mutation(_html()))
    with pytest.raises(ValueError):
        find_gems._handle_bvb_shares(bvb, session)


@pytest.mark.parametrize(
    "failure",
    [
        requests.Timeout,
        requests.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ],
)
def test_fetch_retries_transient_body_or_transport_failure(failure, monkeypatch):
    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)
    session = MagicMock()
    response = _response("ok")
    session.get.side_effect = [failure("transient"), response]
    assert find_gems._fetch_source(session, "https://example.com") is response
    session.get.side_effect = failure("persistent")
    with pytest.raises(failure):
        find_gems._fetch_source(session, "https://example.com")
    assert session.get.call_count == 4


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 429, 501, 505])
def test_fetch_does_not_retry_permanent_http_error(status):
    session = MagicMock()
    response = _response("")
    response.status_code = status
    session.get.return_value = response
    with pytest.raises(requests.HTTPError):
        find_gems._fetch_source(session, "https://example.com")
    assert session.get.call_count == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_fetch_retries_transient_http_status_once(status, monkeypatch):
    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)
    failure = _response("")
    failure.status_code = status
    failure.raw = MagicMock()
    success = _response("ok")
    session = MagicMock()
    session.get.side_effect = [failure, success]
    assert find_gems._fetch_source(session, "https://example.com") is success
    assert session.get.call_count == 2
    failure.raw.close.assert_called_once()
    session.get.reset_mock()
    session.get.side_effect = [failure, failure]
    with pytest.raises(requests.HTTPError):
        find_gems._fetch_source(session, "https://example.com")
    assert session.get.call_count == 2


def test_dependency_check_uses_bs4_import_and_distribution_name(monkeypatch, capsys):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda module: None if module == "bs4" else object()
    )
    find_gems._check_deps()
    output = capsys.readouterr().err
    assert "beautifulsoup4" in output
    assert "poetry add beautifulsoup4" in output


def test_all_configured_paginated_sources_use_page_parameter():
    config = json.loads(Path("config/exchanges.json").read_text())
    sources = [
        ex
        for ex in config["exchanges"]
        if ex.get("params", {}).get("paginate_max_pages", 1) > 1
    ]
    assert len(sources) == 5
    assert all(ex["params"]["page_param"] == "page" for ex in sources)


def test_utf8_without_charset_and_explicit_encoding():
    config = {"source_url": "https://example.com", "params": {"ticker_col": "Symbol"}}
    session = MagicMock()
    html = "<table><tr><th>Symbol</th><th>Name</th></tr><tr><td>BRD</td><td>Société Générale</td></tr></table>"
    session.get.return_value = _response(html)
    assert (
        find_gems._handle_scrape_html(config, session).iloc[0]["Name"]
        == "Société Générale"
    )
    response = _response(html)
    response._content = html.encode("latin-1")
    session.get.return_value = response
    config["params"]["source_encoding"] = "latin-1"
    assert (
        find_gems._handle_scrape_html(config, session).iloc[0]["Name"]
        == "Société Générale"
    )


def test_unique_floor_cannot_be_satisfied_by_duplicates(bvb, monkeypatch):
    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)
    monkeypatch.setitem(
        find_gems._HANDLERS,
        "bvb_shares",
        lambda *_: pd.DataFrame({"Symbol": ["TLV"] * 5, "Company": ["Bank"] * 5}),
    )
    with pytest.raises(RuntimeError, match="got 1"):
        find_gems.scrape_exchanges(
            {"meta": {"description": "test"}, "exchanges": [bvb]}
        )


@pytest.mark.parametrize("scrape_only", [True, False])
def test_failed_source_preserves_existing_output(
    bvb, tmp_path, monkeypatch, scrape_only
):
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"meta": {"description": "test"}, "exchanges": [bvb]}))
    output = tmp_path / "gems.txt"
    output.write_text("OLD.RO\n")
    monkeypatch.setitem(find_gems._HANDLERS, "bvb_shares", lambda *_: pd.DataFrame())
    argv = ["find_gems", "--configfile", str(config), "--output", str(output)]
    if scrape_only:
        argv.append("--scrape-only")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="Required exchange"):
        find_gems.main()
    assert output.read_text() == "OLD.RO\n"


def test_bvb_scrape_to_pipeline_tickers_and_metadata(bvb, tmp_path, monkeypatch):
    from src.ibkr.ticker import Ticker
    from src.macro_regions import get_macro_region_info

    session = MagicMock()
    session.get.return_value = _response(_html())
    monkeypatch.setattr(find_gems, "_get_session", lambda: session)
    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)
    df = find_gems.scrape_exchanges(
        {"meta": {"description": "test"}, "exchanges": [bvb]}
    )
    out = tmp_path / "gems.txt"
    find_gems.write_outputs(df, str(out))
    assert set(out.read_text().splitlines()) == {"TLV.RO", "H2O.RO"}
    for symbol in out.read_text().splitlines():
        info = get_macro_region_info(symbol)
        assert (info.country, info.macro_region) == ("Romania", "EUROPE")
        ticker = Ticker.from_yf(symbol)
        assert ticker.exchange == "BVB"
        assert Ticker.from_ibkr(ticker.symbol, "BVB", "RON").yf == symbol
    assert find_gems._to_usd(1_000_000, "RON", {"RON": 0.2193}) == pytest.approx(
        219_300
    )


def test_configured_exchange_currency_coverage():
    from src.exchange_metadata import SUFFIX_TO_CURRENCY_CODE
    from src.fx_normalization import FALLBACK_RATES_TO_USD

    config = json.loads(Path("config/exchanges.json").read_text())
    for ex in config["exchanges"]:
        if not ex.get("enabled", True) and ex["yahoo_suffix"] != ".RO":
            continue
        suffixes = (
            ex["params"].get("suffix_map", {}).values()
            if ex["yahoo_suffix"] == "dynamic"
            else [ex["yahoo_suffix"]]
        )
        for suffix in suffixes:
            currency = SUFFIX_TO_CURRENCY_CODE.get(suffix, "USD")
            if currency == "GBX":
                currency = "GBP"
            assert currency in {*find_gems._FX_CURRENCIES, "USD"}
            assert currency in FALLBACK_RATES_TO_USD


def test_bvb_ingestion_valuation_identity_and_concentration(monkeypatch):
    from src.ibkr.models import AnalysisRecord, PortfolioSummary
    from src.ibkr.portfolio import normalize_positions
    from src.ibkr.portfolio_health import compute_portfolio_health
    from src.ibkr.reconciler import reconcile
    from src.ibkr.reconciliation_rules import (
        _exchange_from_position,
        analysis_identity_verified,
    )

    cache = MagicMock()
    cache.resolve_rates_sync.return_value = {"RON": (0.2193, "fixture")}
    monkeypatch.setattr("src.ibkr.portfolio.get_fx_rate_cache", lambda: cache)
    raw = {
        "conid": 123456,
        "contractDesc": "TLV",
        "listingExchange": "BVB",
        "currency": "RON",
        "position": 100,
        "mktPrice": 34,
        "avgCost": 30,
        "mktValue": 3400,
        "unrealizedPnl": 400,
    }
    normalized = normalize_positions([raw])
    assert not normalized.unresolved
    pos = normalized.resolved[0]
    assert pos.yf_ticker == "TLV.RO"
    assert pos.currency == "RON"
    assert pos.market_value_usd == pytest.approx(745.62)
    assert pos.market_value_basis == "LOCAL_CONVERTED"
    assert _exchange_from_position(pos) == "RO"
    analysis = AnalysisRecord(
        ticker="TLV.RO", analysis_date="2026-09-15", currency="RON"
    )
    assert analysis_identity_verified(pos, analysis)
    assert not analysis_identity_verified(
        pos, AnalysisRecord(ticker="TLV.T", analysis_date="2026-09-15")
    )
    portfolio = PortfolioSummary(portfolio_value_usd=745.62)
    monkeypatch.setattr("src.ibkr.reconciler._load_structural_macro_events", lambda: [])
    items = reconcile([pos], {"TLV.RO": analysis}, portfolio)
    assert any(item.ticker.yf == "TLV.RO" for item in items)
    assert portfolio.exchange_weights == {"RO": 100.0}
    assert portfolio.currency_weights == {"RON": 100.0}
    flags = compute_portfolio_health([pos], {}, portfolio)
    assert any("RO (Romania)" in flag for flag in flags)


def test_romanian_chart_currency_and_unverified_smart_identity():
    from src.charts.chart_node import _get_currency_format
    from src.ibkr.models import AnalysisRecord, NormalizedPosition
    from src.ibkr.reconciliation_rules import analysis_identity_verified
    from src.ibkr.ticker import Ticker

    assert _get_currency_format("TLV.RO").format_price(34.5).endswith(" lei")
    pos = NormalizedPosition(
        conid=123456,
        ticker=Ticker.from_ibkr("TLV", "SMART", "RON"),
        currency="RON",
        quantity=100,
        ticker_identity_verified=True,
    )
    assert pos.yf_ticker == "TLV.RO"
    assert not analysis_identity_verified(
        pos, AnalysisRecord(ticker="TLV.RO", analysis_date="2026-09-15", currency="RON")
    )


def test_bvb_official_host_and_lookalike_rejection():
    from src.tools.official_documents import is_official_document_url

    assert is_official_document_url("https://www.bvb.ro/infocont/report.pdf")
    assert is_official_document_url("https://iris.bvb.ro/ReportDetails/example")
    assert not is_official_document_url("https://bvb.ro.example.org/report.pdf")


@pytest.mark.parametrize("publishable", [True, False])
def test_romanian_pm_artifact_preserves_identity_and_validity(
    tmp_path, monkeypatch, publishable
):
    from src.persistence import save_results_to_file

    monkeypatch.setattr("src.memory.get_ticker_memory_stats", lambda *_: {})
    monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})
    result = {
        "prediction_snapshot": {
            "ticker": "TLV.RO",
            "analysis_date": "2026-09-15",
            "currency": "RON",
            "verdict": "HOLD",
        },
        "final_trade_decision": "PORTFOLIO MANAGER VERDICT: HOLD\n### --- START PM_BLOCK ---\nVERDICT: HOLD\nCURRENCY: RON\n### --- END PM_BLOCK ---",
        "analysis_validity": {"publishable": publishable},
        "macro_context_report": "Romanian listed bank; European macro context.",
        "run_summary": {"macro_context_region": "EUROPE", "publishable": publishable},
        "structured_inputs": {
            "raw_financial_metrics": {
                "status": "VALID",
                "payload": {"currency": "RON", "currentPrice": 34.0},
            }
        },
    }
    output = save_results_to_file(result, "TLV.RO", results_dir=tmp_path)
    payload = json.loads(output.read_text())
    assert payload["metadata"]["ticker"] == "TLV.RO"
    assert payload["macro_context"]["region"] == "EUROPE"
    assert payload["analysis_validity"]["publishable"] is publishable
    from src.ibkr.analysis_index import _build_analysis_record_from_file

    record = _build_analysis_record_from_file(output)
    assert record is not None
    assert (record.ticker, record.currency, record.verdict) == ("TLV.RO", "RON", "HOLD")
    assert (
        payload["structured_inputs"]["raw_financial_metrics"]["payload"]["currency"]
        == "RON"
    )


def test_overlapping_pages_and_repeated_terminal_page(monkeypatch):
    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)

    def page(symbols):
        return _response(pd.DataFrame({"Symbol": symbols}).to_html(index=False))

    session = MagicMock()
    session.get.side_effect = [page(["A", "B"]), page(["B", "C"]), page(["A", "B"])]
    df = find_gems._handle_scrape_html(
        {
            "source_url": "https://example.com",
            "params": {"ticker_col": "Symbol", "paginate_max_pages": 5},
        },
        session,
    )
    assert df.Symbol.tolist() == ["A", "B", "C"]
    assert session.get.call_count == 3


@pytest.mark.parametrize(
    ("currency", "ticker_symbol", "live_rate"),
    [("RON", "TLV.RO", 0.2193), ("BRL", "PETR4.SA", 0.197)],
)
@pytest.mark.parametrize("use_live_rate", [True, False], ids=["live", "fallback"])
def test_configured_fx_rate_reaches_screening(
    monkeypatch, currency, ticker_symbol, live_rate, use_live_rate
):
    from src.fx_normalization import FALLBACK_RATES_TO_USD

    monkeypatch.setattr(find_gems.time, "sleep", lambda _: None)

    monkeypatch.setattr(
        find_gems,
        "_fetch_one_fx_rate",
        lambda requested_currency: (
            requested_currency,
            live_rate if use_live_rate and requested_currency == currency else None,
        ),
    )
    rates = find_gems._fetch_fx_rates()
    expected = live_rate if use_live_rate else FALLBACK_RATES_TO_USD[currency]
    assert rates[currency] == expected
    ticker = MagicMock(
        info={
            "currency": currency,
            "quoteType": "EQUITY",
            "currentPrice": 10,
            "marketCap": 1_000_000_000,
            "averageVolume": 100_000,
            "trailingPE": 10,
        },
        income_stmt=pd.DataFrame(),
    )
    monkeypatch.setattr(find_gems.yf, "Ticker", lambda _: ticker)
    result = find_gems._process_row(
        {"YF_Ticker": ticker_symbol},
        fx_rates=rates,
        min_mcap=50_000_000,
        min_volume=100_000,
    )
    assert result["Market_Cap_USD"] == pytest.approx(1_000_000_000 * expected)
    assert result["Daily_Turnover_USD"] == pytest.approx(1_000_000 * expected)
    ticker.info["averageVolume"] = 10
    assert (
        find_gems._process_row(
            {"YF_Ticker": ticker_symbol}, fx_rates=rates, min_volume=100_000
        )
        is None
    )
