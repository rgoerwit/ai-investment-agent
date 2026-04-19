from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

from src.ibkr.client import IbkrClient
from src.ibkr.exceptions import IBKRAPIError, IBKRAuthError, IBKRSessionConflictError
from src.ibkr.order_builder import parse_price
from src.ibkr.ticker_mapper import (
    cache_conid_mapping,
    ibkr_symbol_to_yf,
    yf_to_ibkr_format,
)

logger = structlog.get_logger(__name__)

_PROBE_CACHE_TTL_SECONDS = 60.0
_FIELD_LAST_PRICE = "31"
_FIELD_SYMBOL = "55"
_FIELD_BID = "84"
_FIELD_ASK = "86"
_FIELD_VOLUME = "87"
_FIELD_EXCHANGE = "6004"
_FIELD_MARKET_DATA_AVAILABILITY = "6509"
_FIELD_COMPANY_NAME = "7051"


@dataclass
class IbkrSecurityProbe:
    configured: bool
    requested_ticker: str
    identity_confidence: str  # VERIFIED | AMBIGUOUS | UNVERIFIED
    resolved_conid: int | None = None
    resolved_symbol: str | None = None
    resolved_yf_ticker: str | None = None
    company_name: str | None = None
    listing_exchange: str | None = None
    exchange: str | None = None
    currency: str | None = None
    is_tradeable: bool | None = None
    market_data_availability: str | None = None
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    used_brokerage_session: bool = False
    error_kind: str | None = None
    error_message: str | None = None


class IbkrSecurityDataService:
    """Low-volume IBKR probe for identity, mapping, and quote rescue."""

    def __init__(
        self,
        *,
        config=None,
        client_cls: type[IbkrClient] | None = None,
        cache_ttl_secs: float = _PROBE_CACHE_TTL_SECONDS,
    ) -> None:
        self._config = config
        self._client_cls = client_cls or IbkrClient
        self._cache_ttl_secs = cache_ttl_secs
        self._probe_cache: dict[str, tuple[float, IbkrSecurityProbe]] = {}
        self._not_configured_logged = False

    async def probe_security(self, yf_ticker: str) -> IbkrSecurityProbe:
        cached = self._get_cached_probe(yf_ticker)
        if cached is not None:
            return cached

        probe = await asyncio.to_thread(self._probe_security_sync, yf_ticker)
        self._set_cached_probe(yf_ticker, probe)
        return probe

    def _resolve_config(self):
        if self._config is not None:
            return self._config

        from src.ibkr_config import ibkr_config

        return ibkr_config

    def _get_cached_probe(self, yf_ticker: str) -> IbkrSecurityProbe | None:
        cached = self._probe_cache.get(yf_ticker.upper())
        if not cached:
            return None
        expires_at, probe = cached
        if time.monotonic() >= expires_at:
            self._probe_cache.pop(yf_ticker.upper(), None)
            return None
        return probe

    def _set_cached_probe(self, yf_ticker: str, probe: IbkrSecurityProbe) -> None:
        self._probe_cache[yf_ticker.upper()] = (
            time.monotonic() + self._cache_ttl_secs,
            probe,
        )

    def _probe_security_sync(self, yf_ticker: str) -> IbkrSecurityProbe:
        config = self._resolve_config()
        if not config.is_configured():
            if not self._not_configured_logged:
                logger.info("ibkr_security_probe_unavailable", reason="not_configured")
                self._not_configured_logged = True
            return IbkrSecurityProbe(
                configured=False,
                requested_ticker=yf_ticker,
                identity_confidence="UNVERIFIED",
                error_kind="NOT_CONFIGURED",
            )

        client = self._client_cls(config)
        try:
            client.connect(brokerage_session=False)
            symbol, expected_exchange = yf_to_ibkr_format(yf_ticker)
            raw_candidates = client.stock_conid_by_symbol(
                symbol, default_filtering=False
            )
            candidates = self._extract_candidates(raw_candidates, symbol)
            candidate, confidence = self._select_candidate(
                candidates, expected_exchange
            )

            if candidate is None:
                error_kind = "NO_MATCH" if not candidates else "AMBIGUOUS"
                return IbkrSecurityProbe(
                    configured=True,
                    requested_ticker=yf_ticker,
                    identity_confidence=confidence,
                    error_kind=error_kind,
                )

            conid = self._to_int(candidate.get("conid"))
            if conid is None:
                return IbkrSecurityProbe(
                    configured=True,
                    requested_ticker=yf_ticker,
                    identity_confidence="UNVERIFIED",
                    error_kind="NO_MATCH",
                )

            info = client.get_contract_info(conid, compete=False)
            exchange = self._first_non_empty(
                info.get("exchange"),
                candidate.get("exchange"),
            )
            listing_exchange = self._first_non_empty(
                info.get("primaryExch"),
                info.get("listingExchange"),
                exchange,
            )
            currency = self._first_non_empty(
                info.get("currency"),
                candidate.get("currency"),
            )
            resolved_symbol = self._first_non_empty(
                info.get("symbol"),
                candidate.get("symbol"),
                symbol,
            )
            company_name = self._first_non_empty(
                info.get("companyName"),
                info.get("longName"),
                info.get("name"),
            )
            resolved_yf_ticker = ibkr_symbol_to_yf(
                resolved_symbol or symbol,
                listing_exchange or exchange or "",
                currency or "",
            )

            probe = IbkrSecurityProbe(
                configured=True,
                requested_ticker=yf_ticker,
                identity_confidence=confidence,
                resolved_conid=conid,
                resolved_symbol=resolved_symbol,
                resolved_yf_ticker=resolved_yf_ticker,
                company_name=company_name,
                listing_exchange=listing_exchange,
                exchange=exchange,
                currency=currency,
                is_tradeable=self._to_bool(
                    info.get("tradeable"),
                    info.get("isTradeable"),
                ),
                used_brokerage_session=bool(info),
            )

            if confidence != "VERIFIED":
                return probe

            cache_conid_mapping(
                resolved_yf_ticker,
                conid,
                resolved_symbol or symbol,
                listing_exchange or exchange or "",
            )

            snapshot = client.get_marketdata_snapshot(conid, compete=False)
            if snapshot:
                probe.used_brokerage_session = True
                probe.market_data_availability = self._first_non_empty(
                    snapshot.get(_FIELD_MARKET_DATA_AVAILABILITY),
                    probe.market_data_availability,
                )
                probe.company_name = self._first_non_empty(
                    snapshot.get(_FIELD_COMPANY_NAME),
                    probe.company_name,
                )
                probe.resolved_symbol = self._first_non_empty(
                    snapshot.get(_FIELD_SYMBOL),
                    probe.resolved_symbol,
                )
                probe.exchange = self._first_non_empty(
                    snapshot.get(_FIELD_EXCHANGE),
                    probe.exchange,
                )
                probe.last_price = self._snapshot_number(
                    snapshot.get(_FIELD_LAST_PRICE)
                )
                probe.bid = self._snapshot_number(snapshot.get(_FIELD_BID))
                probe.ask = self._snapshot_number(snapshot.get(_FIELD_ASK))
                probe.volume = self._snapshot_number(snapshot.get(_FIELD_VOLUME))
                if probe.resolved_symbol:
                    probe.resolved_yf_ticker = ibkr_symbol_to_yf(
                        probe.resolved_symbol,
                        probe.listing_exchange or probe.exchange or "",
                        probe.currency or "",
                    )

            return probe
        except IBKRSessionConflictError as exc:
            return IbkrSecurityProbe(
                configured=True,
                requested_ticker=yf_ticker,
                identity_confidence="UNVERIFIED",
                error_kind="SESSION_CONFLICT",
                error_message=str(exc),
            )
        except IBKRAuthError as exc:
            return IbkrSecurityProbe(
                configured=True,
                requested_ticker=yf_ticker,
                identity_confidence="UNVERIFIED",
                error_kind="AUTH",
                error_message=str(exc),
            )
        except IBKRAPIError as exc:
            error_text = str(exc)
            return IbkrSecurityProbe(
                configured=True,
                requested_ticker=yf_ticker,
                identity_confidence="UNVERIFIED",
                error_kind="RATE_LIMIT" if "429" in error_text else "API_ERROR",
                error_message=error_text,
            )
        except Exception as exc:
            logger.debug("ibkr_security_probe_failed", ticker=yf_ticker, error=str(exc))
            return IbkrSecurityProbe(
                configured=True,
                requested_ticker=yf_ticker,
                identity_confidence="UNVERIFIED",
                error_kind="API_ERROR",
                error_message=str(exc),
            )
        finally:
            client.close()

    @staticmethod
    def _extract_candidates(raw_candidates: dict[str, Any], symbol: str) -> list[dict]:
        if not isinstance(raw_candidates, dict):
            return []
        for key, value in raw_candidates.items():
            if key.upper() == symbol.upper() and isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _select_candidate(
        candidates: list[dict[str, Any]], expected_exchange: str
    ) -> tuple[dict[str, Any] | None, str]:
        if not candidates:
            return None, "UNVERIFIED"

        exact_matches = [
            candidate
            for candidate in candidates
            if candidate.get("exchange") == expected_exchange
        ]
        if len(exact_matches) == 1:
            return exact_matches[0], "VERIFIED"
        if len(exact_matches) > 1:
            return None, "AMBIGUOUS"
        if len(candidates) == 1 and not expected_exchange:
            return candidates[0], "VERIFIED"
        return None, "AMBIGUOUS"

    @staticmethod
    def _snapshot_number(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # IBKR prefixes numeric fields with "C" (calculated/stale) or "H" (halted).
        if text[:1] in {"C", "H"} and len(text) > 1:
            text = text[1:].strip()
        return parse_price(text)

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_bool(*values: Any) -> bool | None:
        for value in values:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                cleaned = value.strip().lower()
                if cleaned in {"true", "yes", "1"}:
                    return True
                if cleaned in {"false", "no", "0"}:
                    return False
            if isinstance(value, int | float) and value in {0, 1}:
                return bool(value)
        return None

    @staticmethod
    def _first_non_empty(*values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
