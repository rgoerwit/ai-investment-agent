"""Shared test doubles for the retrospective / lessons-learned loop.

These are deliberately *behavioural* fakes rather than ``MagicMock``s. The
retrospective's dedup logic is entirely a ChromaDB ``where`` clause, so a mock
whose ``.get()`` returns a fixed payload passes identically before and after a
change to that clause — it asserts nothing. Likewise the evaluation memo's whole
purpose is to avoid repeat network calls, which is only expressible against a
double that counts them.

Every seam the retrospective touches is faked here: the vector store, yfinance,
and the lesson LLM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB
# ══════════════════════════════════════════════════════════════════════════════

_UNSET = object()


def _matches(meta: Mapping[str, Any], clause: Mapping[str, Any]) -> bool:
    """Evaluate a ChromaDB-style ``where`` clause against one metadata dict.

    Supports the operator subset this repository actually uses: ``$and``, ``$or``,
    ``$eq``, ``$ne``, ``$in``, ``$nin``, plus the bare ``{"key": value}`` shorthand.
    """
    for key, condition in clause.items():
        if key == "$and":
            if not all(_matches(meta, sub) for sub in condition):
                return False
            continue
        if key == "$or":
            if not any(_matches(meta, sub) for sub in condition):
                return False
            continue

        actual = meta.get(key, _UNSET)
        if not isinstance(condition, Mapping):
            if actual is _UNSET or actual != condition:
                return False
            continue

        for operator, expected in condition.items():
            if operator == "$eq":
                if actual is _UNSET or actual != expected:
                    return False
            elif operator == "$ne":
                if actual is not _UNSET and actual == expected:
                    return False
            elif operator == "$in":
                if actual is _UNSET or actual not in expected:
                    return False
            elif operator == "$nin":
                if actual is not _UNSET and actual in expected:
                    return False
            else:  # pragma: no cover - guards against silently ignoring an operator
                raise AssertionError(f"fake does not implement operator {operator!r}")
    return True


class FakeCollection:
    """An in-memory stand-in for a ChromaDB collection."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._next_id = 0
        self.get_calls = 0
        self.query_calls = 0

    # -- writes ----------------------------------------------------------------
    def add(
        self,
        *,
        ids: Sequence[str] | None = None,
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
        embeddings: Any = None,
    ) -> None:
        documents = list(documents or [])
        metadatas = list(metadatas or [{} for _ in documents])
        if ids is None:
            ids = []
            for _ in documents:
                ids.append(f"fake-{self._next_id}")
                self._next_id += 1
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        ):
            self.records.append(
                {"id": record_id, "document": document, "metadata": dict(metadata)}
            )

    def delete(self, *, ids: Iterable[str] | None = None, **_: Any) -> None:
        if ids is None:
            return
        removing = set(ids)
        self.records = [r for r in self.records if r["id"] not in removing]

    # -- reads -----------------------------------------------------------------
    def get(self, where: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        self.get_calls += 1
        matched = [
            r for r in self.records if where is None or _matches(r["metadata"], where)
        ]
        return {
            "ids": [r["id"] for r in matched],
            "documents": [r["document"] for r in matched],
            "metadatas": [dict(r["metadata"]) for r in matched],
        }

    def query(
        self,
        *,
        n_results: int = 10,
        where: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.query_calls += 1
        matched = [
            r for r in self.records if where is None or _matches(r["metadata"], where)
        ]
        matched = matched[:n_results]
        return {
            "ids": [[r["id"] for r in matched]],
            "documents": [[r["document"] for r in matched]],
            "metadatas": [[dict(r["metadata"]) for r in matched]],
            "distances": [[0.1 for _ in matched]],
        }

    def count(self) -> int:
        return len(self.records)


class FakeLessonsMemory:
    """Behavioural double for ``FinancialSituationMemory``."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.situation_collection = FakeCollection()
        self.add_calls = 0

    async def add_situations(
        self,
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> bool:
        if not self.available:
            return False
        self.add_calls += 1
        self.situation_collection.add(documents=documents, metadatas=metadatas)
        return True

    # -- convenience for arranging fixtures ------------------------------------
    def seed(self, document: str = "seeded", **metadata: Any) -> None:
        self.situation_collection.add(documents=[document], metadatas=[metadata])

    def metadatas(self) -> list[dict[str, Any]]:
        return [dict(r["metadata"]) for r in self.situation_collection.records]

    def documents(self) -> list[str]:
        """The stored text. Needed once records became deterministic renderings:
        the text itself is now an assertable contract, not model output."""
        return [str(r["document"]) for r in self.situation_collection.records]


# ══════════════════════════════════════════════════════════════════════════════
# yfinance
# ══════════════════════════════════════════════════════════════════════════════


class FakeTicker:
    def __init__(self, owner: FakeYFinance, symbol: str) -> None:
        self._owner = owner
        self._symbol = symbol

    def history(self, **_: Any) -> pd.DataFrame:
        self._owner.history_calls.append(self._symbol)
        series = self._owner.prices.get(self._symbol)
        if series is None:
            if self._owner.raise_on_missing:
                raise RuntimeError(f"no data for {self._symbol}")
            return pd.DataFrame({"Close": []})
        return pd.DataFrame({"Close": list(series)})

    @property
    def info(self) -> dict[str, Any]:
        self._owner.info_calls.append(self._symbol)
        return self._owner.infos.get(self._symbol, {})


class FakeYFinance:
    """Counts every round-trip, which is the assertion the memo needs.

    ``prices`` maps symbol -> (start_close, end_close). A symbol that is absent
    yields an empty frame (or raises, with ``raise_on_missing``) — that is how a
    missing benchmark is simulated.
    """

    def __init__(
        self,
        prices: Mapping[str, Sequence[float]] | None = None,
        *,
        infos: Mapping[str, Mapping[str, Any]] | None = None,
        raise_on_missing: bool = False,
    ) -> None:
        self.prices = dict(prices or {})
        self.infos = dict(infos or {})
        self.raise_on_missing = raise_on_missing
        self.history_calls: list[str] = []
        self.info_calls: list[str] = []

    def Ticker(self, symbol: str) -> FakeTicker:  # noqa: N802 - mirrors yfinance
        return FakeTicker(self, symbol)

    @property
    def call_count(self) -> int:
        return len(self.history_calls)


# ══════════════════════════════════════════════════════════════════════════════
# Lesson LLM
# ══════════════════════════════════════════════════════════════════════════════


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLessonLLM:
    """Captures the prompt it was handed; Step 8's assertions all read it."""

    DEFAULT_REPLY = (
        "LESSON: Prefer corroborated cash flow over reported cash flow.\n"
        "TYPE: missed_risk\n"
        "FAILURE_MODE: OPERATIONAL_MISS"
    )

    def __init__(self, reply: str | None = None) -> None:
        self.reply = reply if reply is not None else self.DEFAULT_REPLY
        self.prompts: list[str] = []

    async def ainvoke(self, messages: Any, config: Any = None) -> _FakeResponse:
        first = messages[0]
        self.prompts.append(getattr(first, "content", str(first)))
        return _FakeResponse(self.reply)

    @property
    def call_count(self) -> int:
        return len(self.prompts)

    @property
    def last_prompt(self) -> str:
        assert self.prompts, "the lesson LLM was never invoked"
        return self.prompts[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Corpus construction
# ══════════════════════════════════════════════════════════════════════════════


def days_ago(days: int) -> str:
    """Analysis dates are always relative.

    A pinned date in a freshness-gated test is a time bomb: it passes until the
    fixture ages past a threshold it was never meant to exercise. Goldens pin
    dates; behaviour gated on elapsed time derives them.
    """
    return (date.today() - timedelta(days=days)).isoformat()


def make_snapshot(
    ticker: str = "2767.T",
    *,
    age_days: int = 180,
    verdict: str = "BUY",
    analysis_id: str | None = None,
    source_file: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A prediction snapshot with the fields the retrospective actually reads."""
    snapshot: dict[str, Any] = {
        "ticker": ticker,
        "analysis_date": days_ago(age_days),
        "analysis_id": analysis_id,
        "verdict": verdict,
        "current_price": 1000.0,
        "sector": "Industrials",
        "exchange": ticker.split(".")[-1] if "." in ticker else "US",
        "currency": "JPY",
        "benchmark_index": "^N225",
        "fx_rate_to_usd": 0.0067,
        "health_adj": 70.0,
        "growth_adj": 60.0,
        "is_quick_mode": False,
        "is_strict_mode": False,
        "decision_intent": "reasoning",
        "bear_risks_excerpt": "Cyclical exposure to semiconductor capex.",
    }
    snapshot.update(overrides)
    snapshot["_source_file"] = source_file or (
        f"{ticker}_{snapshot['analysis_date'].replace('-', '')}_000000_analysis.json"
    )
    return snapshot


def write_analysis_artifact(
    directory: Any,
    snapshot: Mapping[str, Any],
    *,
    filename: str | None = None,
) -> Any:
    """Write a minimal ``*_analysis.json`` carrying ``snapshot``."""
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or str(snapshot.get("_source_file") or "unnamed_analysis.json")
    payload = {k: v for k, v in snapshot.items() if k != "_source_file"}
    path = directory / name
    with open(path, "w") as handle:
        json.dump({"prediction_snapshot": payload}, handle)
    return path


def yfinance_ticker_stub(*, stock=(100.0, 65.0), benchmark=(100.0, 70.0)):
    """A `yfinance.Ticker` replacement for `patch.object(yfinance, "Ticker", ...)`.

    The seam matters: `src/retrospective.py` does `import yfinance as yf` *inside*
    the fetch function, so `monkeypatch.setattr("src.retrospective.yf", ...)`
    patches an attribute nothing reads. A test doing that reaches the live
    network and passes or fails on real market data — which two tests written on
    2026-08-17 did, undetected, because their tickers exist.
    """
    import pandas as pd

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_: object):
            series = benchmark if self.symbol.startswith("^") else stock
            if series is None:
                return pd.DataFrame({"Close": []})
            return pd.DataFrame({"Close": list(series)})

        @property
        def info(self) -> dict:
            return {}

    return _Ticker
