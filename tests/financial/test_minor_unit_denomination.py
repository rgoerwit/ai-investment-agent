"""A currency code is the unit — never discard it, never guess it from a venue.

GAMA.L raised a CAPITAL ALLOCATION REVIEW at `price 976.98 >= reference 11.50`
on a stock that had not moved: an ~85x unit mismatch producing a false action
signal. yfinance was consistent (every LSE quote is `GBp`, in pence) and the
fetcher was correct (it converts to `GBP` only when marketCap/PE corroborate,
and stamps the resulting code either way). The label was then destroyed
downstream and re-guessed from the ticker suffix.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from src.fx_normalization import (
    MINOR_UNIT_CURRENCY_ALIASES,
    canonical_currency_code,
    get_fx_rate_fallback,
    normalize_minor_unit_currency,
)

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


class TestCanonicalCurrencyCode:
    def test_minor_unit_codes_survive_canonicalization(self):
        """`"GBp".upper()` is `"GBP"` — a 100x different currency."""
        assert canonical_currency_code("GBp") == "GBp"
        assert canonical_currency_code("GBX") == "GBX"

    def test_ordinary_codes_still_upper_case(self):
        assert canonical_currency_code("usd") == "USD"
        assert canonical_currency_code("  mxn  ") == "MXN"

    def test_lowercase_major_is_not_captured_by_the_minor_alias(self):
        """`"gbp"` means pounds; only the exact `"GBp"` spelling means pence."""
        assert canonical_currency_code("gbp") == "GBP"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_empty_input_is_none(self, value):
        assert canonical_currency_code(value) is None


class TestMinorUnitRoundTrip:
    """The property that makes a new venue a table edit, not a code path."""

    @pytest.mark.parametrize("minor", sorted(MINOR_UNIT_CURRENCY_ALIASES))
    def test_minor_and_major_describe_the_same_money(self, minor):
        major, scale = normalize_minor_unit_currency(minor)
        # 100 minor units == 1 major unit, whatever the venue.
        assert pytest.approx(100 * scale) == 1.0
        assert normalize_minor_unit_currency(major) == (major, 1.0)

    @pytest.mark.parametrize("minor", sorted(MINOR_UNIT_CURRENCY_ALIASES))
    def test_fx_resolves_the_minor_code_directly(self, minor):
        """No conversion to the major unit is required to reach USD."""
        major, scale = normalize_minor_unit_currency(minor)
        minor_rate = get_fx_rate_fallback(minor)
        major_rate = get_fx_rate_fallback(major)
        assert minor_rate is not None and major_rate is not None
        assert minor_rate == pytest.approx(major_rate * scale, rel=1e-6)


class TestNoVenueConditionalUnitArithmetic:
    """The shape of the removed bug, and the one most likely to return.

    Scaling a price by 100 inside a branch keyed on a ticker suffix or exchange
    name is the `.L -> x100` rule: right only when the fetcher happened to
    decline a conversion, and blind to which case it is in. Matched on the AST,
    not on text — a prose description of the retired pattern is not the pattern.
    """

    @staticmethod
    def _mentions_venue(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                value = child.value
                if value.startswith(".") and 2 <= len(value) <= 4:
                    return True
            if isinstance(child, ast.Attribute) and child.attr in {
                "suffix",
                "exchange",
                "listingExchange",
            }:
                return True
        return False

    @staticmethod
    def _scales_by_hundred(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign | ast.BinOp) and isinstance(
                getattr(child, "op", None), ast.Mult | ast.Div
            ):
                operand = getattr(child, "value", None) or getattr(child, "right", None)
                if isinstance(operand, ast.Constant) and operand.value in (100, 0.01):
                    return True
        return False

    # The one legitimate venue-keyed scale in the codebase. LSE *history*
    # (not quotes) carries no currency label at all, so there is no code to ask
    # — documented at src/liquidity_calculation_tool.py::_resolve_turnover_price.
    # Every other path reaches a labelled quote and must convert by code.
    _ALLOWED = {
        "src/liquidity_calculation_tool.py": "lse_history_has_no_currency_label"
    }

    def test_no_module_scales_a_price_under_a_venue_branch(self):
        offenders: list[str] = []
        for path in _SRC.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                if self._mentions_venue(node.test) and any(
                    self._scales_by_hundred(stmt) for stmt in node.body
                ):
                    rel = str(path.relative_to(_SRC.parent))
                    if rel in self._ALLOWED:
                        continue
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "venue-conditional unit arithmetic reintroduced at "
            f"{offenders}. Convert by currency code (normalize_minor_unit_currency) "
            "so an already-major value scales by 1.0 instead of 100x."
        )


class TestCurrencyCodesAreNotCaseFlattened:
    """`.upper()` on a currency code silently destroys `GBp`."""

    @pytest.mark.parametrize(
        "module", ["currency_resolver.py", "ibkr/analysis_index.py"]
    )
    def test_resolution_boundaries_do_not_upper_case_currency(self, module):
        source = (_SRC / module).read_text()
        offenders = [
            line.strip()
            for line in source.splitlines()
            if ".upper()" in line
            and re.search(r"currency|_ccy", line)
            and "canonical_currency_code" not in line
            # Comparing already-canonical codes for equality is fine; only
            # *producing* a code via .upper() destroys the denomination.
            and not line.lstrip().startswith("#")
            and "==" not in line
        ]
        assert not offenders, (
            f"{module} upper-cases a currency code: {offenders}. "
            "Use canonical_currency_code, which preserves minor-unit aliases."
        )


class TestPriceCurrencyIsCodeOwned:
    """The model declares the field; the code decides its value.

    A model-authored denomination would satisfy the contract while replacing
    evidence with transcription — the failure the guidance-contract incident
    already taught. ``stamp_price_currency`` therefore *overwrites*, never
    backfills, so no LLM behaviour can make the unit of record wrong.
    """

    BODY = "TICKER: GAMA.L\nCURRENT_PRICE: 975.5\nPE_RATIO_TTM: 14.1"

    @staticmethod
    def _stamped(body: str, payload: dict) -> str | None:
        from src.agents.fundamentals_reconciler import stamp_price_currency

        for line in stamp_price_currency(body, payload).splitlines():
            if line.startswith("PRICE_CURRENCY:"):
                return line.split(":", 1)[1].strip()
        return None

    @pytest.mark.parametrize(
        "model_wrote",
        [
            "GBP",  # plausible-but-wrong: the exact 100x error
            "pence",  # prose instead of a code
            "£",  # a symbol
            "N/A",  # gave up
            "GBp (pence)",  # annotated
            "",  # emitted the key with no value
            "USD",  # a different economy entirely
        ],
    )
    def test_model_written_value_is_always_overwritten(self, model_wrote):
        body = f"{self.BODY}\nPRICE_CURRENCY: {model_wrote}"
        assert self._stamped(body, {"currency": "GBp"}) == "GBp"

    def test_field_is_added_when_the_model_omits_it_entirely(self):
        """LLMs drop fields. The contract must not depend on compliance."""
        assert self._stamped(self.BODY, {"currency": "GBp"}) == "GBp"

    def test_absent_payload_currency_is_explicit_not_inherited(self):
        """An unknown unit reads as unknown rather than the model's guess.

        Note the caller gates this on a structured payload existing at all: with
        no payload the sanitizer leaves the block untouched, because stamping
        N/A there would destroy a possibly-correct transcription and add a line
        to every thin-data block. Consumers take the unit from the
        payload-derived snapshot currency, so an unstamped block simply falls
        back to suffix resolution.
        """
        body = f"{self.BODY}\nPRICE_CURRENCY: GBP"
        assert self._stamped(body, {}) == "N/A"

    def test_a_venue_changing_denomination_needs_no_code_change(self):
        """If the LSE ever quoted in pounds, the payload would simply say so.

        Nothing keys on the venue, so a redenomination (or a provider switching
        convention) is followed automatically rather than requiring a rule edit.
        """
        assert self._stamped(self.BODY, {"currency": "GBP"}) == "GBP"
        assert self._stamped(self.BODY, {"currency": "GBp"}) == "GBp"

    def test_a_brand_new_minor_unit_venue_is_a_table_edit(self):
        """ZAc/ILA are not registered yet: they stamp through unharmed.

        Registering one later changes only MINOR_UNIT_CURRENCY_ALIASES — the
        stamp, the resolver and the comparison all read the table.
        """
        assert self._stamped(self.BODY, {"currency": "ZAc"}) == "ZAC"
        assert "ZAc" not in MINOR_UNIT_CURRENCY_ALIASES

    def test_stamp_leaves_every_other_line_untouched(self):
        from src.agents.fundamentals_reconciler import stamp_price_currency

        out = stamp_price_currency(self.BODY, {"currency": "GBp"})
        for line in self.BODY.splitlines():
            assert line in out


class TestTradeBlockInheritsTheDenomination:
    """ENTRY/STOP/TARGET_* are derived from the DATA_BLOCK price.

    The Trader prompt asks it to copy PRICE_CURRENCY across, but a copy must
    never *be* the unit of record: these are the levels the reconciler later
    compares against a live position price, and a 100x error there is the
    GAMA.L false valuation-reference review.
    """

    FUNDAMENTALS = (
        "### --- START DATA_BLOCK ---\n"
        "CURRENT_PRICE: 975.5\n"
        "PRICE_CURRENCY: GBp\n"
        "### --- END DATA_BLOCK ---"
    )
    TRADE_BLOCK = (
        "TRADE_BLOCK:\nACTION: BUY\nENTRY: 950\nSTOP: 880\n"
        "TARGET_1: 1150\nTARGET_2: 1300\nHORIZON: 12 months"
    )

    @staticmethod
    def _stamped(content: str, fundamentals: str) -> str | None:
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        out = stamp_trade_block_price_currency(content, fundamentals)
        for line in out.splitlines():
            if line.strip().startswith("PRICE_CURRENCY:"):
                return line.split(":", 1)[1].strip()
        return None

    @pytest.mark.parametrize(
        "model_wrote", ["GBP", "pounds sterling", "N/A", "USD", ""]
    )
    def test_trader_transcription_is_overwritten(self, model_wrote):
        content = f"{self.TRADE_BLOCK}\nPRICE_CURRENCY: {model_wrote}"
        assert self._stamped(content, self.FUNDAMENTALS) == "GBp"

    def test_field_is_inserted_when_the_trader_omits_it(self):
        assert self._stamped(self.TRADE_BLOCK, self.FUNDAMENTALS) == "GBp"

    def test_insertion_preserves_every_existing_level(self):
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        out = stamp_trade_block_price_currency(self.TRADE_BLOCK, self.FUNDAMENTALS)
        for line in self.TRADE_BLOCK.splitlines():
            assert line in out

    @pytest.mark.parametrize(
        "fundamentals",
        [
            "no data block at all",
            "### --- START DATA_BLOCK ---\nCURRENT_PRICE: 9\n### --- END DATA_BLOCK ---",
            "### --- START DATA_BLOCK ---\nPRICE_CURRENCY: N/A\n### --- END DATA_BLOCK ---",
        ],
    )
    def test_unknown_denomination_is_left_unstated(self, fundamentals):
        """Better silent than asserting a unit nobody established."""
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        assert (
            stamp_trade_block_price_currency(self.TRADE_BLOCK, fundamentals)
            == self.TRADE_BLOCK
        )

    def test_missing_anchor_line_does_not_corrupt_the_block(self):
        """A malformed TRADE_BLOCK is returned intact, never half-edited."""
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        malformed = "TRADE_BLOCK:\nACTION: BUY\nHORIZON: 12 months"
        assert (
            stamp_trade_block_price_currency(malformed, self.FUNDAMENTALS) == malformed
        )


class TestStampsSurviveMisbehavingModels:
    """Flash-tier models under --quick do not reliably follow format rules.

    Every case here was executed against the stamps before being written down;
    the injection and duplicate cases were live defects found this way.
    """

    FUND = (
        "### --- START DATA_BLOCK ---\n"
        "CURRENT_PRICE: 975.5\nPRICE_CURRENCY: GBp\n"
        "### --- END DATA_BLOCK ---"
    )
    TB = "TRADE_BLOCK:\nENTRY: 950\nTARGET_2: 1300\nHORIZON: 12"

    @staticmethod
    def _codes(text: str) -> list[str]:
        return [
            line.split(":", 1)[1].strip().strip("*")
            for line in text.splitlines()
            if "PRICE_CURRENCY" in line
        ]

    @pytest.mark.parametrize("hostile", ["GB\\1p", "\\g<0>X", "GBp\\", "(GBp)"])
    def test_regex_template_in_a_value_cannot_inject_or_raise(self, hostile):
        """A value is data, never a replacement template.

        `re.sub` interprets `\\1` and `\\g<0>` in its replacement argument, so an
        upstream value containing them used to raise inside the trader node
        (failing the artifact) or splice text into the block.
        """
        from src.agents.fundamentals_reconciler import (
            stamp_price_currency,
            stamp_trade_block_price_currency,
        )

        fundamentals = self.FUND.replace("GBp", hostile)
        # Not a currency code => the block is left alone rather than corrupted.
        assert stamp_trade_block_price_currency(self.TB, fundamentals) == self.TB
        assert self._codes(stamp_price_currency("X: 1", {"currency": hostile})) == [
            "N/A"
        ]

    def test_duplicate_fields_are_made_consistent_not_contradictory(self):
        """A model repeating the key must not leave two different answers.

        Both lines are rewritten, so a parser taking the first or the last gets
        the same code. (Two identical lines is untidy but unambiguous.)
        """
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        content = f"{self.TB}\nPRICE_CURRENCY: GBP\nPRICE_CURRENCY: USD"
        assert set(
            self._codes(stamp_trade_block_price_currency(content, self.FUND))
        ) == {"GBp"}

    def test_markdown_emphasis_around_the_field_is_still_replaced(self):
        """`**PRICE_CURRENCY: GBP**` is a routine flash-model output."""
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        content = "TRADE_BLOCK:\n**PRICE_CURRENCY: GBP**\nTARGET_2: 1300"
        assert self._codes(stamp_trade_block_price_currency(content, self.FUND)) == [
            "GBp"
        ]

    def test_only_the_first_trade_block_is_stamped_when_a_model_repeats_itself(self):
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        content = f"{self.TB}\n\nTRADE_BLOCK:\nTARGET_2: 99\n"
        assert self._codes(stamp_trade_block_price_currency(content, self.FUND)) == [
            "GBp"
        ]

    @pytest.mark.parametrize("content", ["", "no trade block at all", "TRADE_BLOCK:"])
    def test_content_without_an_anchor_is_returned_unchanged(self, content):
        """No TARGET_2 line means no block to annotate — never a partial edit."""
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        assert stamp_trade_block_price_currency(content, self.FUND) == content

    def test_a_bare_anchor_still_receives_the_stamp(self):
        """A TARGET_2 line with no value is still a real anchor."""
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        assert self._codes(
            stamp_trade_block_price_currency("TARGET_2:", self.FUND)
        ) == ["GBp"]

    def test_crlf_content_is_not_mangled(self):
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        out = stamp_trade_block_price_currency(self.TB.replace("\n", "\r\n"), self.FUND)
        assert self._codes(out) == ["GBp"]
        assert "ENTRY: 950" in out

    @pytest.mark.parametrize(
        "junk", ["pounds sterling", "£", "N/A", "GBP (pence)", "1234", "G"]
    )
    def test_non_code_values_never_reach_the_trade_block(self, junk):
        from src.agents.fundamentals_reconciler import (
            stamp_trade_block_price_currency,
        )

        fundamentals = self.FUND.replace("GBp", junk)
        assert stamp_trade_block_price_currency(self.TB, fundamentals) == self.TB
