"""No ordinary retrospective outcome reaches a generator.

The failures of August 2026 were not disobedience. The prompt asked, for an
unexplained residual with only price data, *"write what should be CHECKED next
time"* — a question with no grounded answer, so the model supplied one: FX
exposure for a bear case that never mentions currency (2PP.DE), liquidity and
momentum screens for one about analyst coverage and margins (3008.TW). Four
rounds of rules tried to constrain the output; none could, because the request
itself was impossible.

Templates make those outputs *unrepresentable* rather than prohibited. A review
record quotes what was recorded and stops; there is no slot for a mechanism the
record does not contain, and no slot for "rather than trust fundamentals less".

Two kinds of assertion here, and both are needed. The first is that the generator
is never awaited — a contract about the pipeline. The second is that the rendered
text cannot contain the specific inventions — a contract about the templates.
"""

from __future__ import annotations

import json

import pytest

from src.lesson_disposition import (
    DRIVER_MARKET,
    DRIVER_RESIDUAL,
    HYPOTHESIS_TOPIC_BEAR_CASE,
    HYPOTHESIS_TOPIC_KILL_CRITERIA,
    HYPOTHESIS_TOPIC_RED_FLAGS,
    DispositionVerdict,
    EvidenceCapability,
    LessonDisposition,
    OutcomeFacts,
    derive_disposition,
    render_record,
)
from src.retrospective import (
    UNRESOLVED_PRICE_ONLY,
    CachedRegimeDelta,
    _deterministic_lesson_type,
    _hypothesis_for_review,
    build_lesson_record,
    disposition_for,
    run_retrospective,
)
from tests.advanced.retrospective_fakes import (
    FakeLessonsMemory,
    make_snapshot,
    yfinance_ticker_stub,
)

# 2PP.DE as recorded: residual-dominated, bear case present, no regime.
_2PP_DE = {
    "ticker": "2PP.DE",
    "verdict": "DO_NOT_INITIATE",
    "days_elapsed": 173,
    "price_return_pct": 12.0,
    "benchmark_return_pct": -14.1,
    "excess_return_pct": 26.1,
    "benchmark_used": "^GDAXI",
    "bear_risks_excerpt": (
        "*   **US Revenue**: 56.88% (Threshold: <35%) - **HARD FAIL**\n"
        "*   **Eroding Competitive Moat**: Branded checkout is losing share."
    ),
    "attribution": {
        "dominant_driver": DRIVER_RESIDUAL,
        "market_return_pct": -14.1,
        "residual_return_pct": 26.1,
    },
    "cached_regime_delta": {"shifted": None},
}

# 3008.TW: evidence covers coverage, growth, margins, governance — never
# liquidity, momentum or anything technical.
_3008_TW = {
    **_2PP_DE,
    "ticker": "3008.TW",
    "verdict": "HOLD",
    "excess_return_pct": 67.5,
    "bear_risks_excerpt": (
        "1. Analyst coverage is heavy. 2. Growth has decelerated. "
        "3. Structural margin erosion. 4. Governance concentration."
    ),
}

# The words each lesson invented, which no template may be able to emit.
_2PP_DE_INVENTIONS = ("fx", "currency", "sector momentum")
_3008_TW_INVENTIONS = ("liquidity", "momentum", "technical", "breakout")

HYPOTHESIS_ONLY = frozenset({EvidenceCapability.HYPOTHESIS})
BOTH_CAPS = frozenset({EvidenceCapability.HYPOTHESIS, EvidenceCapability.CONTEXT})


class TestTheGeneratorIsNeverAwaited:
    """A contract about the pipeline, asserted for every ordinary disposition."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("label", "comparison"),
        [
            ("2PP.DE residual with a bear case", _2PP_DE),
            ("3008.TW residual with a bear case", _3008_TW),
            (
                "market-dominated but no regime recorded",
                {**_2PP_DE, "attribution": {"dominant_driver": DRIVER_MARKET}},
            ),
            (
                "active tender",
                {**_2PP_DE, "m_and_a_status": "ACTIVE_TENDER"},
            ),
            (
                "withheld: context only, unattributed",
                {
                    **_2PP_DE,
                    "bear_risks_excerpt": "",
                    "regime_at_decision": {"risk_appetite": "RISK_ON"},
                },
            ),
            (
                "skipped: no evidence at all",
                {**_2PP_DE, "bear_risks_excerpt": ""},
            ),
        ],
    )
    async def test_no_disposition_calls_the_lesson_llm(
        self, label, comparison, monkeypatch
    ):
        called: list[object] = []

        async def _forbidden(*args, **kwargs):
            called.append(args)
            raise AssertionError(f"{label} reached the generator")

        monkeypatch.setattr("src.retrospective.generate_lesson", _forbidden)
        build_lesson_record(comparison, disposition_for(comparison))
        assert called == []


class TestTheProbeFailuresAreUnrepresentable:
    """A contract about the templates, not about the model's cooperation."""

    def test_2pp_de_cannot_mention_fx_or_momentum(self):
        record = build_lesson_record(_2PP_DE, disposition_for(_2PP_DE))
        assert record is not None
        text = record[0].lower()
        for invention in _2PP_DE_INVENTIONS:
            assert invention not in text, f"template emitted {invention!r}"

    def test_3008_tw_cannot_mention_liquidity_or_technical_screens(self):
        record = build_lesson_record(_3008_TW, disposition_for(_3008_TW))
        assert record is not None
        text = record[0].lower()
        for invention in _3008_TW_INVENTIONS:
            assert invention not in text, f"template emitted {invention!r}"

    def test_no_template_can_demote_the_screen_by_contrast(self):
        """The other gate criterion: no "rather than" subordinating fundamentals."""
        facts = OutcomeFacts(
            ticker="T.T",
            days_elapsed=173,
            price_return_pct=-20.0,
            benchmark_return_pct=-5.0,
            excess_return_pct=-15.0,
            benchmark_used="^N225",
            market_return_pct=-5.0,
            residual_return_pct=-15.0,
            regime_label="RISK_ON / NONE",
        )
        for disposition in LessonDisposition:
            if not disposition.produces_record:
                continue
            # Codes paired with the disposition that actually emits them: a
            # contextual reason on a review disposition is not a combination the
            # policy can produce, and rendering now refuses it outright.
            codes = {
                LessonDisposition.REVIEW_HYPOTHESIS: (
                    "review:non_market:residual",
                    "review:no_regime_recorded",
                    "review:regime_comparison_unknown",
                    "review:regime_shifted",
                ),
                LessonDisposition.CONTEXTUAL_OBSERVATION: (
                    "market_dominated_stable_regime",
                ),
                LessonDisposition.SPECIAL_SITUATION_REVIEW: ("active_tender",),
            }[disposition]
            for reason_code in codes:
                text = render_record(
                    DispositionVerdict(disposition, reason_code, "because"),
                    facts,
                    hypothesis="1. Debt load.",
                    hypothesis_topic="x",
                ).lower()
                for banned in ("rather than", "instead of", "not merely"):
                    assert banned not in text

    def test_the_recorded_hypothesis_is_quoted_verbatim(self):
        """No paraphrase: the retrospective cannot test the claim, so it must
        not restate it in its own words."""
        record = build_lesson_record(_2PP_DE, disposition_for(_2PP_DE))
        assert record is not None
        assert "56.88%" in record[0]
        assert "Eroding Competitive Moat" in record[0]

    def test_materialization_is_always_declared_untested(self):
        record = build_lesson_record(_2PP_DE, disposition_for(_2PP_DE))
        assert record is not None
        assert "NOT_EVALUATED" in record[0]


class TestWithheldOutcomesProduceNothing:
    @pytest.mark.parametrize(
        "comparison",
        [
            {
                **_2PP_DE,
                "bear_risks_excerpt": "",
                "regime_at_decision": {"risk_appetite": "RISK_ON"},
            },
            {**_2PP_DE, "bear_risks_excerpt": ""},
        ],
        ids=["context-only unattributed", "no evidence"],
    )
    def test_no_record_is_built(self, comparison):
        assert build_lesson_record(comparison, disposition_for(comparison)) is None


class TestDeterministicFieldsReplaceModelJudgment:
    def test_the_hypothesis_source_is_reported_not_interpreted(self):
        """`hypothesis_topic` names where the claim came from.

        Asking a model for a "topic" is how a causal vocabulary re-enters after
        being removed from the failure-mode field.
        """
        assert _hypothesis_for_review(_2PP_DE)[1] == HYPOTHESIS_TOPIC_BEAR_CASE
        assert (
            _hypothesis_for_review(
                {**_2PP_DE, "kill_criteria": ["Margin below 8% for two quarters"]}
            )[1]
            == HYPOTHESIS_TOPIC_KILL_CRITERIA
        )
        assert (
            _hypothesis_for_review(
                {
                    **_2PP_DE,
                    "bear_risks_excerpt": "",
                    "red_flags_at_decision": ["CMIC_FLAGGED"],
                }
            )[1]
            == HYPOTHESIS_TOPIC_RED_FLAGS
        )

    def test_a_pre_registered_trigger_outranks_the_bear_prose(self):
        """The sharpest decision-time claim wins: it was falsifiable in advance."""
        text, topic = _hypothesis_for_review(
            {**_2PP_DE, "kill_criteria": ["Margin below 8%"]}
        )
        assert topic == HYPOTHESIS_TOPIC_KILL_CRITERIA
        assert text == "Margin below 8%"

    @pytest.mark.parametrize(
        ("verdict", "excess", "expected"),
        [
            ("BUY", -30.0, "false_positive"),
            ("BUY", 40.0, "correct_call"),
            ("DO_NOT_INITIATE", 30.0, "missed_opportunity"),
            ("DO_NOT_INITIATE", -30.0, "correct_call"),
            ("HOLD", 40.0, "missed_opportunity"),
        ],
    )
    def test_lesson_type_is_derived_not_asked(self, verdict, excess, expected):
        """Prediction against price is a fact.

        Asking a model for it added nothing but the chance of a different answer.
        """
        assert (
            _deterministic_lesson_type(
                {"verdict": verdict, "excess_return_pct": excess}
            )
            == expected
        )

    def test_failure_mode_never_asserts_a_cause(self):
        record = build_lesson_record(_2PP_DE, disposition_for(_2PP_DE))
        assert record is not None
        assert record[2] == UNRESOLVED_PRICE_ONLY, (
            "no deterministic path establishes a cause, so the causal vocabulary "
            "must never be populated"
        )


class TestTheGeneratorIsNeverAwaitedByThePipeline:
    """The contract the unit test above cannot establish.

    `build_lesson_record` is synchronous and structurally incapable of calling an
    LLM, so asserting that it does not is close to vacuous — it validates the
    renderer, not the routing. These drive `run_retrospective` itself with the
    generator patched to raise, which is the only way to prove that no path
    through the orchestrator reaches it.
    """

    def _snapshot(self, tmp_path, name, **overrides):
        snapshot = make_snapshot(age_days=180, analysis_id=name)
        snapshot.update(
            {
                "ticker": "T.T",
                "verdict": "BUY",
                "bear_risks_excerpt": "1. Cyclical exposure at a peak.",
                "benchmark_index": "^N225",
            }
        )
        snapshot.update(overrides)
        return snapshot

    async def _run(self, tmp_path, snapshot, monkeypatch, memory):
        async def _forbidden(*_a, **_k):
            raise AssertionError("run_retrospective reached the lesson generator")

        monkeypatch.setattr("src.retrospective.generate_lesson", _forbidden)
        # `import yfinance as yf` lives INSIDE the fetch function, so
        # `src.retrospective.yf` is not a seam — patching it silently does
        # nothing and the call goes to the live network.
        import yfinance

        monkeypatch.setattr(
            yfinance,
            "Ticker",
            yfinance_ticker_stub(stock=(100.0, 60.0), benchmark=(100.0, 98.0)),
        )
        seen: list = []
        await run_retrospective(
            None,
            tmp_path,
            lessons_memory=memory,
            memo_path=tmp_path / "memo.json",
            on_summary=seen.append,
        )
        return seen[-1]

    def _write(self, tmp_path, snapshot):
        (tmp_path / "T.T_20260101_000000_analysis.json").write_text(
            json.dumps({"prediction_snapshot": snapshot})
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("label", "overrides"),
        [
            ("review hypothesis", {}),
            (
                "special situation",
                {"m_and_a_status": "ACTIVE_TENDER"},
            ),
            (
                "withheld: context only",
                {
                    "bear_risks_excerpt": "",
                    "regime_at_decision": {"risk_appetite": "RISK_ON"},
                },
            ),
        ],
    )
    async def test_no_route_reaches_the_generator(
        self, tmp_path, monkeypatch, label, overrides
    ):
        memory = FakeLessonsMemory()
        self._write(tmp_path, self._snapshot(tmp_path, "run-a", **overrides))
        summary = await self._run(tmp_path, None, monkeypatch, memory)
        assert summary.reconciles, label

    @pytest.mark.asyncio
    async def test_a_withheld_outcome_writes_nothing_and_is_accounted_for(
        self, tmp_path, monkeypatch
    ):
        """No Chroma record, no generated lesson, but the run still explains it."""
        memory = FakeLessonsMemory()
        self._write(
            tmp_path,
            self._snapshot(
                tmp_path,
                "run-a",
                bear_risks_excerpt="",
                regime_at_decision={"risk_appetite": "RISK_ON"},
            ),
        )
        summary = await self._run(tmp_path, None, monkeypatch, memory)

        assert memory.add_calls == 0, "a withheld outcome must write no record"
        assert summary.withheld_no_record == 1
        assert summary.generated == 0
        assert summary.stored == 0
        assert summary.reconciles

        recorded = json.loads((tmp_path / "memo.json").read_text())
        assert any(
            entry["outcome"] == "WITHHELD_NO_RECORD" for entry in recorded.values()
        ), "the pricing was paid for; it must not be repeated in 30 days"

    @pytest.mark.asyncio
    async def test_a_review_outcome_does_write_a_deterministic_record(
        self, tmp_path, monkeypatch
    ):
        """The counterpart: withholding must not be achieved by writing nothing ever."""
        memory = FakeLessonsMemory()
        self._write(tmp_path, self._snapshot(tmp_path, "run-a"))
        summary = await self._run(tmp_path, None, monkeypatch, memory)
        assert summary.stored == 1
        assert memory.add_calls == 1
        stored_text = memory.documents()[0]
        assert stored_text.startswith("REVIEW —")
        assert "NOT_EVALUATED" in stored_text


class TestTheGeneratorHasNoProductionCaller:
    """`generate_lesson` is reserved, not supported.

    It still holds the free-form prompt that produced the inventions. Retained as
    the starting point for a future evidence-backed post-mortem — the only
    context where a generator has something to synthesize — but nothing may wire
    it into the ordinary path. Same shape as the `risk_reward_ratio` guard.
    """

    def test_nothing_in_src_calls_it(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        callers = []
        for path in (root / "src").rglob("*.py"):
            text = path.read_text()
            for match in re.finditer(r"\bgenerate_lesson\s*\(", text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line = text[line_start : text.find("\n", match.start())]
                if line.lstrip().startswith(("def ", "async def ")):
                    continue
                callers.append(f"{path.relative_to(root)}: {line.strip()}")
        assert callers == [], (
            f"generate_lesson is reserved for a future evidence-backed "
            f"post-mortem and must not be on the ordinary path: {callers}"
        )


class TestEachDispositionProducesItsOwnRecordShape:
    """Asserting the run *completed* proves nothing about which record it wrote.

    An earlier version of this file parametrized a case labelled "contextual
    observation" and checked only `summary.reconciles`. It produced a
    REVIEW_HYPOTHESIS: with no macro cache the regime delta is `None`, and the
    fixture's prices were residual-dominated anyway. The label was aspirational
    and the assertion could not tell.
    """

    def _write(self, tmp_path, **overrides):
        snapshot = make_snapshot(age_days=180, analysis_id="run-a")
        snapshot.update(
            {
                "ticker": "T.T",
                "verdict": "BUY",
                "bear_risks_excerpt": "1. Cyclical exposure at a peak.",
                "benchmark_index": "^N225",
            }
        )
        snapshot.update(overrides)
        (tmp_path / "T.T_20260101_000000_analysis.json").write_text(
            json.dumps({"prediction_snapshot": snapshot})
        )

    async def _run(self, tmp_path, monkeypatch, memory, *, prices, stable_regime=False):
        import yfinance

        async def _forbidden(*_a, **_k):
            raise AssertionError("the pipeline reached the lesson generator")

        monkeypatch.setattr("src.retrospective.generate_lesson", _forbidden)
        monkeypatch.setattr(yfinance, "Ticker", yfinance_ticker_stub(**prices))
        if stable_regime:
            # The delta is resolved from an on-disk macro cache that a tmp_path
            # run does not have, so without this the regime is `None` and the
            # contextual path is unreachable — which is what the old test hit.
            monkeypatch.setattr(
                "src.retrospective.resolve_cached_regime_delta",
                lambda *_a, **_k: CachedRegimeDelta(
                    shifted=False, shift_reason="no change in risk appetite or shock"
                ),
            )
        await run_retrospective(
            None, tmp_path, lessons_memory=memory, memo_path=tmp_path / "memo.json"
        )

    @pytest.mark.asyncio
    async def test_a_stable_market_outcome_renders_a_contextual_observation(
        self, tmp_path, monkeypatch
    ):
        """The only injectable disposition, and the only one never reached before.

        The prices are load-bearing and hard to get right, which is the point:
        a lesson triggers on |excess| (= the residual) clearing 15% for a BUY,
        while MARKET dominance needs the market leg 1.5x larger *still*. So
        -40% market against a -20% residual, i.e. the stock at -60% and the index
        at -40%. A first attempt used -38% against -2% and stored nothing at all,
        because a 2% excess never triggers — the same arithmetic that makes
        market-dominated outcomes genuinely rare in the corpus.
        """
        memory = FakeLessonsMemory()
        self._write(tmp_path, regime_at_decision={"risk_appetite": "RISK_ON"})
        await self._run(
            tmp_path,
            monkeypatch,
            memory,
            prices={"stock": (100.0, 40.0), "benchmark": (100.0, 60.0)},
            stable_regime=True,
        )
        assert memory.add_calls == 1
        meta = memory.metadatas()[0]
        assert meta["lesson_disposition"] == "CONTEXTUAL_OBSERVATION"
        assert meta["lesson_eligibility"] == "INJECTABLE", (
            "the sole injectable path; if this regresses nothing can ever inject"
        )
        assert memory.documents()[0].startswith("CONTEXT —")
        assert meta["hypothesis_topic"] == "", (
            "a contextual observation does not consult the bear case, so stamping "
            "its source would imply evidence the record never used"
        )

    @pytest.mark.asyncio
    async def test_an_active_tender_renders_a_special_situation_record(
        self, tmp_path, monkeypatch
    ):
        memory = FakeLessonsMemory()
        self._write(tmp_path, m_and_a_status="ACTIVE_TENDER")
        await self._run(
            tmp_path,
            monkeypatch,
            memory,
            prices={"stock": (100.0, 60.0), "benchmark": (100.0, 98.0)},
        )
        assert memory.add_calls == 1
        assert memory.metadatas()[0]["lesson_disposition"] == "SPECIAL_SITUATION_REVIEW"
        text = memory.documents()[0]
        assert text.startswith("SPECIAL SITUATION —")
        assert "not diagnostic" in text


class TestTheMarketDominatedReviewWordingIsHonest:
    """The contradiction fixed in B's reason codes, then recreated in C's template.

    `REVIEW_HYPOTHESIS` covers *both* an unattributed outcome and a
    market-dominated one whose regime cannot authorize an observation. A fixed
    preamble said "not attributed to the market" of the latter.
    """

    @pytest.mark.parametrize(
        ("shifted", "expected_fragment"),
        [
            (None, "no decision-time regime was recorded"),
            (False, "no decision-time regime was recorded"),
        ],
    )
    def test_market_with_no_regime_says_so(self, shifted, expected_fragment):
        comparison = {
            **_2PP_DE,
            "attribution": {"dominant_driver": DRIVER_MARKET},
            "cached_regime_delta": {"shifted": shifted},
        }
        record = build_lesson_record(comparison, disposition_for(comparison))
        assert record is not None
        assert "market-dominated" in record[0]
        assert expected_fragment in record[0]
        assert "not attributed to the market" not in record[0]

    def test_a_shifted_regime_says_that_instead(self):
        comparison = {
            **_2PP_DE,
            "attribution": {"dominant_driver": DRIVER_MARKET},
            "regime_at_decision": {"risk_appetite": "RISK_ON"},
            "cached_regime_delta": {"shifted": True},
        }
        record = build_lesson_record(comparison, disposition_for(comparison))
        assert record is not None
        assert "the regime shifted afterwards" in record[0]
        assert "not attributed to the market" not in record[0]

    def test_a_genuinely_unattributed_outcome_still_says_so(self):
        record = build_lesson_record(_2PP_DE, disposition_for(_2PP_DE))
        assert record is not None
        assert "The outcome is not attributed to the market." in record[0]

    def test_an_unknown_blocker_raises_rather_than_defaulting(self):
        """Fail loudly. `.get(code, NON_MARKET)` would hand the contradictory
        sentence to any future market-blocking reason — the same defect again."""
        facts = OutcomeFacts(
            ticker="T.T",
            days_elapsed=173,
            price_return_pct=-20.0,
            benchmark_return_pct=-5.0,
            excess_return_pct=-15.0,
            benchmark_used="^N225",
        )
        with pytest.raises(ValueError, match="no review preamble"):
            render_record(
                DispositionVerdict(
                    LessonDisposition.REVIEW_HYPOTHESIS, "review:some_future_reason", ""
                ),
                facts,
            )

    def test_every_blocker_the_policy_emits_has_a_preamble(self):
        """The completeness guard, so the raise above stays unreachable in practice."""
        for capabilities, driver, shifted in (
            (HYPOTHESIS_ONLY, DRIVER_RESIDUAL, None),
            (HYPOTHESIS_ONLY, DRIVER_MARKET, None),
            (BOTH_CAPS, DRIVER_MARKET, None),
            (BOTH_CAPS, DRIVER_MARKET, True),
        ):
            verdict = derive_disposition(
                capabilities, dominant_driver=driver, regime_shifted=shifted
            )
            render_record(
                verdict,
                OutcomeFacts("T.T", 173, -20.0, -5.0, -15.0, "^N225"),
                hypothesis="1. Debt.",
            )


class TestNoTestPatchesTheWrongYfinanceSeam:
    """A repo-wide guard, because fixing this by hand missed a file.

    `src/retrospective.py` does `import yfinance as yf` *inside* the fetch
    function, so `monkeypatch.setattr("src.retrospective.yf", ...)` patches an
    attribute nothing reads. Worse, the idiom that makes it silent is
    `raising=False` — the one signal that would have reported the missing
    attribute, switched off.

    A test doing this reaches the live network and passes or fails on real market
    data. Three files acquired it on 2026-08-17; two were fixed by hand and the
    third was found only when someone ran the suite offline. The tell was timing:
    the affected test was the slowest in the suite at ~2.4s and dropped to 0.15s
    once stubbed.
    """

    def test_no_test_file_patches_src_retrospective_yf(self):
        """AST, never text.

        A text scan flags this file and `retrospective_fakes.py`, whose
        docstrings quote the retired idiom in order to explain it — the same
        false positive the minor-unit currency guard hit on its own docstring.
        Only an actual call expression counts.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2]
        offenders = []
        for path in (root / "tests").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None
                )
                if name not in {"setattr", "patch", "object"}:
                    continue
                for argument in node.args:
                    if (
                        isinstance(argument, ast.Constant)
                        and argument.value == "src.retrospective.yf"
                    ):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
        assert offenders == [], (
            "these tests patch an attribute that does not exist and therefore "
            f"reach the live network: {offenders}. Use "
            "`monkeypatch.setattr(yfinance, 'Ticker', yfinance_ticker_stub(...))`."
        )
