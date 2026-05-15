from __future__ import annotations

import json

from scripts.summarize_quick_slow_tail import collect_slow_tail_rows, format_summary


def test_collect_slow_tail_rows_orders_by_timeout_loss(tmp_path):
    (tmp_path / "FAST_analysis.json").write_text(
        json.dumps({"token_usage": {"call_diagnostics": {"timeout_seconds_lost": 0}}})
    )
    (tmp_path / "SLOW_analysis.json").write_text(
        json.dumps(
            {
                "ticker": "SLOW",
                "token_usage": {
                    "call_diagnostics": {
                        "timeout_seconds_lost": 300.4,
                        "consultant_timeout": False,
                        "slowest_call": {
                            "agent_name": "Portfolio Manager",
                            "provider": "google",
                            "failure_origin": "provider_sdk_timeout",
                            "elapsed_seconds": 300.4,
                        },
                    }
                },
            }
        )
    )
    (tmp_path / "CONSULTANT_analysis.json").write_text(
        json.dumps(
            {
                "ticker": "CONSULTANT",
                "token_usage": {
                    "call_diagnostics": {
                        "timeout_seconds_lost": 60.0,
                        "consultant_timeout": True,
                        "slowest_call": {
                            "agent_name": "External Consultant",
                            "provider": "openai",
                            "failure_origin": "hard_timeout",
                            "elapsed_seconds": 60.0,
                        },
                    }
                },
            }
        )
    )

    rows = collect_slow_tail_rows(tmp_path)

    assert [row["ticker"] for row in rows] == ["SLOW", "CONSULTANT"]
    assert rows[0]["slowest_origin"] == "provider_sdk_timeout"


def test_format_summary_is_compact():
    summary = format_summary(
        [
            {
                "ticker": "SLOW",
                "timeout_seconds_lost": 300.0,
                "slowest_agent": "Portfolio Manager",
                "slowest_provider": "google",
                "slowest_origin": "provider_sdk_timeout",
                "slowest_elapsed_seconds": 300.0,
                "consultant_timeout": False,
            }
        ],
        limit=1,
    )

    assert summary.splitlines()[0] == (
        "quick_slow_tail_summary count=1 timeout_seconds_lost=300.0"
    )
    assert "ticker=SLOW" in summary
    assert "origin=provider_sdk_timeout" in summary
