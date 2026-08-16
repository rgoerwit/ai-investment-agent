"""Why capital-structure preflight does not use the shared ``URL_RE``.

This is a deliberate divergence, recorded so it is not "consolidated" away later.

``_collect_preflight_outcomes`` scans ``outcome.render()`` output -- Python-rendered tool
results, which routinely embed URLs inside single-quoted strings
(``{'url': 'https://ir.example.co.jp/a'}``). The canonical ``text_patterns.URL_RE``
excludes only ``<>"``, so it captures the closing ``'`` as part of the URL. The candidates
harvested here are passed straight to the official-document tool, where a trailing quote
is a fetch failure, not a cosmetic defect.

Everything else about the two patterns is identical, so this file pins the one property
that justifies keeping them apart.
"""

from __future__ import annotations

import pytest

from src.agents.capital_structure import _URL_IN_QUOTED_PAYLOAD_RE
from src.text_patterns import URL_RE


class TestApostropheExclusionIsLoadBearing:
    @pytest.mark.parametrize(
        "payload",
        [
            "{'url': 'https://ir.example.co.jp/a', 'ok': True}",
            "see 'https://ir.example.co.jp/a' for detail",
            "[('source', 'https://ir.example.co.jp/a')]",
        ],
        ids=["dict_repr", "quoted_prose", "tuple_repr"],
    )
    def test_single_quoted_payload_yields_a_fetchable_url(self, payload: str) -> None:
        found = _URL_IN_QUOTED_PAYLOAD_RE.findall(payload)
        assert found == ["https://ir.example.co.jp/a"]
        assert not found[0].endswith("'")

    @pytest.mark.parametrize(
        ("payload", "over_captured"),
        [
            (
                "{'url': 'https://ir.example.co.jp/a', 'ok': True}",
                "https://ir.example.co.jp/a',",
            ),
            (
                "see 'https://ir.example.co.jp/a' for detail",
                "https://ir.example.co.jp/a'",
            ),
        ],
        ids=["dict_repr", "quoted_prose"],
    )
    def test_shared_url_re_would_capture_the_closing_quote(
        self, payload: str, over_captured: str
    ) -> None:
        """The reason this call site keeps its own pattern -- do not swap it.

        Note the dict-repr case also drags in the trailing comma: ``URL_RE`` excludes
        only ``<>"``, so everything up to the next whitespace comes along.
        """
        assert URL_RE.findall(payload) == [over_captured]


class TestOtherwiseIdenticalToTheSharedPattern:
    @pytest.mark.parametrize(
        "payload",
        [
            "SOURCE_URL: https://ir.example.co.jp/2026/q1.pdf",
            'href="https://ir.example.co.jp/2026/q1.pdf"',
            "<https://ir.example.co.jp/2026/q1.pdf>",
            "trailing prose https://ir.example.co.jp/2026/q1.pdf here",
        ],
        ids=["bare", "double_quoted", "angle_bracketed", "in_prose"],
    )
    def test_agrees_with_shared_pattern_when_no_apostrophe_is_present(
        self, payload: str
    ) -> None:
        assert _URL_IN_QUOTED_PAYLOAD_RE.findall(payload) == URL_RE.findall(payload)

    def test_both_keep_legitimate_parentheses_in_a_path(self) -> None:
        """URLs may legally contain parentheses; neither pattern may truncate there."""
        payload = "https://en.wikipedia.org/wiki/Foo_(bar)"
        assert _URL_IN_QUOTED_PAYLOAD_RE.findall(payload) == [payload]
        assert URL_RE.findall(payload) == [payload]

    def test_an_apostrophe_inside_a_path_is_the_accepted_cost(self) -> None:
        """Truncating at a path apostrophe is the deliberate trade.

        Measured 2 such URLs across 600 artifacts, against single-quoted rendering on
        effectively every preflight payload -- so the exclusion wins on volume. Pinned so
        the trade is visible rather than surprising.
        """
        payload = "https://example.com/o'brien/report.pdf"
        assert _URL_IN_QUOTED_PAYLOAD_RE.findall(payload) == ["https://example.com/o"]
        assert URL_RE.findall(payload) == [payload]
