"""Official documents: a failed retrieval says why, to whom it spoke, and who we were.

A transport failure used to collapse into one ``DOCUMENT_RETRIEVAL_FAILED`` (11 of
~1,619 network attempts in saved runs: TLS certificate faults, a server
disconnect, a read timeout), ``DOCUMENT_HTTP_ERROR`` dropped the status, and the
client sent no User-Agent (``www.sec.gov``: 0 successes in 262 calls). Here every
reason comes from a planted exception below httpx (``MockTransport``), the status
and the log carry what the model and the operator need, the one
``HTTP_USER_AGENT`` reaches every client, and EDGAR requests carry the contact
address or are refused before anything connects. Nothing touches the network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import ssl
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src import editor_tools
from src.config import Settings, config
from src.exchange_metadata import registered_official_document_hosts
from src.stocktwits_api import StockTwitsAPI
from src.tools import official_documents as documents

_SEC_URL = "https://www.sec.gov/Archives/edgar/data/1/report.pdf"
_ISSUER_HOST = registered_official_document_hosts()[0]
_ISSUER_URL = f"https://{_ISSUER_HOST}/disclosure/report.pdf"
_REAL_ASYNC_CLIENT = httpx.AsyncClient
_GUARD_SECONDS = 5.0
_CONTACT = "operator@example.invalid"
_DEFAULT_UA = Settings.model_fields["http_user_agent"].default


class _LogRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def record(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return record

    def failures(self) -> list[dict[str, Any]]:
        return [f for _, e, f in self.events if e == "official_document_failed"]


@pytest.fixture
def logs(monkeypatch) -> _LogRecorder:
    recorder = _LogRecorder()
    monkeypatch.setattr(documents, "logger", recorder)
    return recorder


@pytest.fixture(autouse=True)
def _dns(monkeypatch) -> list[str]:
    """Stub the public-address check; record each URL it was asked about."""
    asked: list[str] = []

    async def public(url: str) -> None:
        asked.append(url)

    monkeypatch.setattr(documents, "_ensure_public_hostname", public)
    return asked


class _Wire:
    """A fake transport below httpx: records every request, answers by handler."""

    def __init__(self, monkeypatch, handler: Callable[[httpx.Request], Any]) -> None:
        self.requests: list[httpx.Request] = []

        async def recording(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            result = handler(request)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        def factory(*, timeout: float, **kwargs: Any) -> httpx.AsyncClient:
            del kwargs
            return _REAL_ASYNC_CLIENT(
                timeout=timeout, transport=httpx.MockTransport(recording)
            )

        monkeypatch.setattr(documents.httpx, "AsyncClient", factory)


def _raises(error: BaseException) -> Callable[[httpx.Request], Any]:
    def handler(request: httpx.Request) -> Any:
        del request
        raise error

    return handler


def _wrapping(outer: BaseException, cause: BaseException) -> BaseException:
    """*outer* as the transport raises it: caused by the OS or ssl error."""
    outer.__cause__ = cause
    return outer


def _ok(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(200, headers={"content-type": "application/octet-stream"})


async def _fetch(url: str = _ISSUER_URL) -> str:
    return await asyncio.wait_for(
        documents.get_official_document.ainvoke({"url": url}), _GUARD_SECONDS
    )


def _reply(reason: str, *extra: str) -> str:
    return "\n".join(["STATUS: INSUFFICIENT_DATA", f"REASON: {reason}", *extra])


# --- Every transport failure has its own reason ------------------------------


_CERT_ERROR = ssl.SSLCertVerificationError("certificate verify failed")
_TRANSPORT_CASES = [
    (httpx.ConnectTimeout("connect timed out"), "DOCUMENT_DOWNLOAD_TIMEOUT"),
    (httpx.ReadTimeout("read timed out"), "DOCUMENT_DOWNLOAD_TIMEOUT"),
    (httpx.PoolTimeout("pool exhausted"), "DOCUMENT_DOWNLOAD_TIMEOUT"),
    (TimeoutError("stdlib timeout"), "DOCUMENT_DOWNLOAD_TIMEOUT"),
    (
        _wrapping(httpx.ConnectError("tls"), _CERT_ERROR),
        "DOCUMENT_TLS_FAILED",
    ),
    (
        _wrapping(httpx.ConnectError("tls"), ssl.SSLError("handshake failure")),
        "DOCUMENT_TLS_FAILED",
    ),
    (
        _wrapping(httpx.ConnectError("refused"), ConnectionRefusedError()),
        "DOCUMENT_CONNECT_FAILED",
    ),
    (httpx.ConnectError("no route"), "DOCUMENT_CONNECT_FAILED"),
    (httpx.RemoteProtocolError("server disconnected"), "DOCUMENT_PROTOCOL_ERROR"),
    (httpx.ReadError("connection reset"), "DOCUMENT_PROTOCOL_ERROR"),
    (httpx.LocalProtocolError("bad request line"), "DOCUMENT_PROTOCOL_ERROR"),
    (httpx.DecodingError("bad gzip"), "DOCUMENT_PROTOCOL_ERROR"),
    (ValueError("unexpected"), "DOCUMENT_RETRIEVAL_FAILED"),
    (RuntimeError("unexpected"), "DOCUMENT_RETRIEVAL_FAILED"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    _TRANSPORT_CASES,
    ids=lambda case: case if isinstance(case, str) else type(case).__name__,
)
async def test_each_planted_transport_failure_has_its_own_reason(
    monkeypatch, error, reason
):
    wire = _Wire(monkeypatch, _raises(error))

    assert await _fetch() == _reply(reason)
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_a_stall_at_the_total_bound_is_a_download_timeout(monkeypatch):
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 0.1)
    never = asyncio.Event()
    wire = _Wire(monkeypatch, lambda request: never.wait())

    assert await _fetch() == _reply("DOCUMENT_DOWNLOAD_TIMEOUT")
    assert len(wire.requests) == 1
    assert not never.is_set()


def _spent(label: str):
    def refuse(seconds: float, *, label: str = label) -> float:
        raise TimeoutError(f"{label!r}: deadline already passed")

    return refuse


@pytest.mark.asyncio
async def test_a_spent_allowance_is_refused_before_any_connection(monkeypatch, _dns):
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 1e-9)
    monkeypatch.setattr(
        documents, "require_allowance", _spent("official_document_download")
    )
    wire = _Wire(monkeypatch, _ok)

    assert await _fetch() == _reply("DOCUMENT_ALLOWANCE_SPENT")
    assert wire.requests == []
    assert _dns == []


@pytest.mark.asyncio
async def test_a_failure_after_the_download_is_never_a_transport_reason(monkeypatch):
    """The reason follows the phase: an ssl or timeout error raised while
    extracting is not the network's."""
    _Wire(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html>x</html>"
        ),
    )
    monkeypatch.setattr(
        documents,
        "_html_document_text",
        MagicMock(side_effect=ssl.SSLError("not a transport failure here")),
    )

    assert await _fetch() == _reply("DOCUMENT_RETRIEVAL_FAILED")


# --- The mapping, on its own --------------------------------------------------


def test_the_cause_chain_is_inspected_for_the_root():
    deep = httpx.ConnectError("connect")
    deep.__cause__ = OSError("socket")
    deep.__cause__.__cause__ = ssl.SSLCertVerificationError("expired")
    assert documents.transport_failure_reason(deep) == "DOCUMENT_TLS_FAILED"

    context_only = httpx.ReadError("reset")
    context_only.__context__ = TimeoutError("inner")
    assert documents.transport_failure_reason(context_only) == (
        "DOCUMENT_DOWNLOAD_TIMEOUT"
    )


def test_a_cyclic_cause_chain_terminates():
    first, second = httpx.ConnectError("a"), OSError("b")
    first.__cause__, second.__cause__ = second, first
    assert documents.transport_failure_reason(first) == "DOCUMENT_CONNECT_FAILED"


def test_every_typed_reason_is_distinct_and_the_residual_is_not_in_the_table():
    reasons = [reason for reason, _ in documents._TRANSPORT_REASONS]
    assert len(reasons) == len(set(reasons))
    assert documents._RESIDUAL_RETRIEVAL_REASON not in reasons


# --- HTTP status ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 500, 429, 400, 599])
async def test_an_http_error_carries_its_status_in_the_output_and_the_log(
    monkeypatch, logs, status
):
    _Wire(monkeypatch, lambda request: httpx.Response(status))

    assert await _fetch() == _reply("DOCUMENT_HTTP_ERROR", f"HTTP_STATUS: {status}")
    (failure,) = logs.failures()
    assert failure["http_status"] == status
    assert failure["reason"] == "DOCUMENT_HTTP_ERROR"


@pytest.mark.asyncio
async def test_a_399_is_not_an_http_error(monkeypatch):
    _Wire(monkeypatch, lambda request: httpx.Response(304))

    assert "DOCUMENT_HTTP_ERROR" not in await _fetch()


# --- Redirects -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        f"http://{_ISSUER_HOST}/disclosure/report.pdf",
        f"http://{_ISSUER_HOST}:80/disclosure/report.pdf",
        "http://www.sec.gov/Archives/report.pdf",
    ],
)
async def test_an_https_to_http_redirect_is_an_insecure_redirect(monkeypatch, target):
    wire = _Wire(
        monkeypatch, lambda request: httpx.Response(302, headers={"location": target})
    )

    assert await _fetch() == _reply("DOCUMENT_INSECURE_REDIRECT")
    assert len(wire.requests) == 1  # refused, never followed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "http://evil.example/report.pdf",
        "https://evil.example/report.pdf",
        f"http://{_ISSUER_HOST}:8080/report.pdf",
        "ftp://www.sec.gov/report.pdf",
    ],
)
async def test_a_redirect_off_the_approved_hosts_stays_unapproved(monkeypatch, target):
    wire = _Wire(
        monkeypatch, lambda request: httpx.Response(302, headers={"location": target})
    )

    assert await _fetch() == _reply("UNAPPROVED_DOCUMENT_HOST")
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_an_unparseable_redirect_location_is_a_protocol_error(monkeypatch):
    """httpx rejects the Location itself, before the tool sees a redirect."""
    wire = _Wire(
        monkeypatch,
        lambda request: httpx.Response(
            302, headers={"location": f"https://{_ISSUER_HOST}:notaport/x"}
        ),
    )

    assert await _fetch() == _reply("DOCUMENT_PROTOCOL_ERROR")
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_a_plain_http_request_url_is_unapproved_not_an_insecure_redirect(
    monkeypatch,
):
    wire = _Wire(monkeypatch, _ok)

    result = await _fetch(f"http://{_ISSUER_HOST}/report.pdf")

    assert result.startswith(_reply("UNAPPROVED_DOCUMENT_HOST"))
    assert wire.requests == []


@pytest.mark.asyncio
async def test_an_https_redirect_between_approved_hosts_is_followed(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == _ISSUER_HOST:
            return httpx.Response(
                302, headers={"location": "https://www.sec.gov/Archives/x.pdf"}
            )
        return httpx.Response(200, headers={"content-type": "application/octet-stream"})

    wire = _Wire(monkeypatch, handler)

    assert await _fetch() == _reply("UNSUPPORTED_DOCUMENT_TYPE")
    assert [r.url.host for r in wire.requests] == [_ISSUER_HOST, "www.sec.gov"]


# --- The failure log ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_log_names_scheme_host_phase_and_reason_but_not_the_url(
    monkeypatch, logs
):
    secret_url = f"https://{_ISSUER_HOST}/private/path/report.pdf?token=SECRET123"
    _Wire(
        monkeypatch,
        _raises(
            _wrapping(
                httpx.ConnectError(f"failed for {secret_url}"),
                ssl.SSLCertVerificationError("certificate verify failed"),
            )
        ),
    )

    await _fetch(secret_url)

    (failure,) = logs.failures()
    assert failure["document_scheme"] == "https"
    assert failure["document_host"] == _ISSUER_HOST
    assert failure["phase"] == "connect"
    assert failure["reason"] == "DOCUMENT_TLS_FAILED"
    rendered = repr(failure)
    assert "SECRET123" not in rendered
    assert "private/path" not in rendered


@pytest.mark.asyncio
async def test_the_log_reports_the_redirect_hop_not_the_requested_url(
    monkeypatch, logs
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == _ISSUER_HOST:
            return httpx.Response(
                302, headers={"location": "https://www.sec.gov/Archives/x.pdf"}
            )
        raise httpx.ReadError("reset")

    _Wire(monkeypatch, handler)

    await _fetch()

    (failure,) = logs.failures()
    assert failure["document_host"] == "www.sec.gov"
    assert failure["phase"] == "connect"


@pytest.mark.asyncio
async def test_phase_dns(monkeypatch, logs):
    async def refuse(url: str) -> None:
        raise documents.DocumentExtractionError("DOCUMENT_DNS_FAILED")

    monkeypatch.setattr(documents, "_ensure_public_hostname", refuse)
    wire = _Wire(monkeypatch, _ok)

    assert await _fetch() == _reply("DOCUMENT_DNS_FAILED")
    assert wire.requests == []
    assert logs.failures()[0]["phase"] == "dns"


@pytest.mark.asyncio
async def test_phase_redirect(monkeypatch, logs):
    _Wire(monkeypatch, lambda request: httpx.Response(302))

    assert await _fetch() == _reply("DOCUMENT_REDIRECT_INVALID")
    assert logs.failures()[0]["phase"] == "redirect"


@pytest.mark.asyncio
async def test_phase_body(monkeypatch, logs):
    small = dataclasses.replace(
        documents.AuditorBudgetPolicy.from_settings(), max_document_bytes=10
    )
    monkeypatch.setattr(
        documents.AuditorBudgetPolicy, "from_settings", classmethod(lambda cls: small)
    )
    _Wire(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 50))

    assert await _fetch() == _reply("DOCUMENT_SIZE_LIMIT")
    assert logs.failures()[0]["phase"] == "body"


@pytest.mark.asyncio
async def test_phase_extract(monkeypatch, logs):
    _Wire(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}),
    )

    assert await _fetch() == _reply("DOCUMENT_TEXT_UNAVAILABLE")
    assert logs.failures()[0]["phase"] == "extract"


@pytest.mark.asyncio
async def test_phase_inspect(monkeypatch, logs):
    _Wire(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"figures " * 100
        ),
    )
    service = MagicMock()
    service.check_within = AsyncMock(return_value=None)
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: service)

    assert await _fetch() == _reply("INSPECTION_TIMEOUT")
    assert logs.failures()[0]["phase"] == "inspect"


@pytest.mark.asyncio
async def test_the_residual_reason_is_logged_with_the_exception_summary(
    monkeypatch, logs
):
    _Wire(monkeypatch, _raises(ValueError("surprise")))

    assert await _fetch() == _reply("DOCUMENT_RETRIEVAL_FAILED")
    (failure,) = logs.failures()
    assert failure["error_type"] == "ValueError"
    assert failure["operation"] == "official_document_failed"


# --- One User-Agent, from one setting ----------------------------------------------


def test_the_default_user_agent_is_the_editor_s_former_constant():
    assert _DEFAULT_UA == (
        "Mozilla/5.0 (compatible; InvestorAgent/1.0; "
        "+https://github.com/rgoerwit/ai-investment-agent)"
    )
    assert not hasattr(editor_tools, "USER_AGENT")


@pytest.mark.parametrize("bad", ["", "   ", "Agent/1.0\r\nX-Injected: 1", "a\x00b"])
def test_a_user_agent_that_cannot_be_a_header_value_is_rejected(bad):
    with pytest.raises(ValueError, match="HTTP_USER_AGENT"):
        Settings(_env_file=None, HTTP_USER_AGENT=bad)


def test_the_env_override_is_read_and_stripped(monkeypatch):
    monkeypatch.setenv("HTTP_USER_AGENT", "  Acme/2.0 (+https://acme.example)  ")
    assert (
        Settings(_env_file=None).http_user_agent == "Acme/2.0 (+https://acme.example)"
    )


@pytest.mark.asyncio
async def test_the_document_client_sends_the_setting(monkeypatch):
    monkeypatch.setattr(config, "http_user_agent", "Doc/1.0 (+https://doc.example)")
    wire = _Wire(monkeypatch, _ok)

    await _fetch()

    assert [r.headers["user-agent"] for r in wire.requests] == [
        "Doc/1.0 (+https://doc.example)"
    ]


class _RecordingSession:
    """Stands in for ``aiohttp.ClientSession``; keeps the headers it was sent."""

    def __init__(self, sent: list[dict[str, str]]) -> None:
        self.sent = sent

    async def __aenter__(self) -> _RecordingSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def get(self, url: str, *, headers: dict[str, str], **kwargs: Any) -> Any:
        del url, kwargs
        self.sent.append(headers)
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={"messages": []})
        manager = AsyncMock()
        manager.__aenter__.return_value = response
        manager.__aexit__.return_value = None
        return manager


async def _stocktwits_headers() -> list[dict[str, str]]:
    sent: list[dict[str, str]] = []
    with patch("aiohttp.ClientSession", return_value=_RecordingSession(sent)):
        await StockTwitsAPI().get_sentiment("AAPL")
    return sent


async def _editor_headers(monkeypatch) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<p>" + "x" * 300
        )

    def factory(*, timeout: float, **kwargs: Any) -> httpx.AsyncClient:
        del kwargs
        return _REAL_ASYNC_CLIENT(
            timeout=timeout, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(editor_tools.httpx, "AsyncClient", factory)
    await editor_tools.fetch_reference_content.ainvoke(
        {"url": "https://reference.example/article"}
    )
    return seen


@pytest.mark.asyncio
async def test_stocktwits_and_the_editor_send_the_setting(monkeypatch):
    monkeypatch.setattr(config, "http_user_agent", "Shared/3.1 (+https://x.example)")

    assert await _stocktwits_headers() == [{"User-Agent": config.http_user_agent}]
    requests = await _editor_headers(monkeypatch)
    assert [r.headers["user-agent"] for r in requests] == [config.http_user_agent]


@pytest.mark.asyncio
async def test_one_env_override_reaches_all_three_clients(monkeypatch):
    monkeypatch.setenv("HTTP_USER_AGENT", "Env/9.9 (+https://env.example)")
    monkeypatch.setattr(
        config, "http_user_agent", Settings(_env_file=None).http_user_agent
    )
    wire = _Wire(monkeypatch, _ok)

    await _fetch()
    (stocktwits,) = await _stocktwits_headers()
    editor = await _editor_headers(monkeypatch)

    assert wire.requests[0].headers["user-agent"] == "Env/9.9 (+https://env.example)"
    assert stocktwits["User-Agent"] == "Env/9.9 (+https://env.example)"
    assert editor[0].headers["user-agent"] == "Env/9.9 (+https://env.example)"


# --- SEC / EDGAR -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sec_request_carries_the_product_token_and_the_contact(monkeypatch):
    monkeypatch.setattr(config, "sec_contact_email", _CONTACT)
    monkeypatch.setattr(config, "http_user_agent", _DEFAULT_UA)
    wire = _Wire(monkeypatch, _ok)

    await _fetch(_SEC_URL)

    assert [r.headers["user-agent"] for r in wire.requests] == [
        f"InvestorAgent/1.0 {_CONTACT}"
    ]


@pytest.mark.asyncio
async def test_the_sec_form_follows_a_custom_user_agent_setting(monkeypatch):
    monkeypatch.setattr(config, "sec_contact_email", _CONTACT)
    monkeypatch.setattr(config, "http_user_agent", "Acme/2.5 (+https://acme.example)")
    wire = _Wire(monkeypatch, _ok)

    await _fetch("https://data.sec.gov/submissions/CIK1.json")

    assert wire.requests[0].headers["user-agent"] == f"Acme/2.5 {_CONTACT}"


@pytest.mark.asyncio
async def test_sec_without_a_contact_is_refused_before_connecting(
    monkeypatch, logs, _dns
):
    monkeypatch.setattr(config, "sec_contact_email", None)
    wire = _Wire(monkeypatch, _ok)

    result = await _fetch(_SEC_URL)

    assert result.splitlines()[:2] == [
        "STATUS: INSUFFICIENT_DATA",
        "REASON: DOCUMENT_SEC_CONTACT_REQUIRED",
    ]
    assert "SEC_CONTACT_EMAIL" in result.splitlines()[2]
    assert wire.requests == []
    assert _dns == []
    assert logs.failures()[0]["phase"] == "dns"


@pytest.mark.asyncio
async def test_a_non_sec_host_never_receives_the_contact(monkeypatch):
    monkeypatch.setattr(config, "sec_contact_email", _CONTACT)
    wire = _Wire(monkeypatch, _ok)

    await _fetch(_ISSUER_URL)

    sent = wire.requests[0].headers["user-agent"]
    assert sent == config.http_user_agent
    assert _CONTACT not in sent


@pytest.mark.asyncio
async def test_a_non_sec_host_needs_no_contact(monkeypatch):
    monkeypatch.setattr(config, "sec_contact_email", None)
    wire = _Wire(monkeypatch, _ok)

    assert await _fetch(_ISSUER_URL) == _reply("UNSUPPORTED_DOCUMENT_TYPE")
    assert len(wire.requests) == 1


@pytest.mark.asyncio
async def test_a_redirect_from_sec_to_another_host_drops_the_contact(monkeypatch):
    monkeypatch.setattr(config, "sec_contact_email", _CONTACT)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.sec.gov":
            return httpx.Response(302, headers={"location": _ISSUER_URL})
        return _ok(request)

    wire = _Wire(monkeypatch, handler)

    await _fetch(_SEC_URL)

    sec, issuer = (r.headers["user-agent"] for r in wire.requests)
    assert _CONTACT in sec
    assert issuer == config.http_user_agent


@pytest.mark.parametrize(
    "url",
    [
        "https://sec.gov.evil.com/x",
        "https://notsec.gov/x",
        "https://evilsec.gov/x",
        "https://sec.gov.example/x",
        "https://example.com/sec.gov",
        "https://example.com/?q=www.sec.gov",
        "https://user@evil.example/sec.gov",
    ],
)
def test_a_lookalike_host_is_not_sec(monkeypatch, url):
    monkeypatch.setattr(config, "sec_contact_email", None)

    # Unset contact would raise for SEC; a lookalike is sent the plain setting.
    assert documents._user_agent_for(url) == config.http_user_agent


@pytest.mark.parametrize(
    "url",
    [
        "https://sec.gov/x",
        "https://www.sec.gov/x",
        "https://data.sec.gov/x",
        "https://WWW.SEC.GOV./x",
    ],
)
def test_sec_and_its_subdomains_require_the_contact(monkeypatch, url):
    monkeypatch.setattr(config, "sec_contact_email", None)
    with pytest.raises(documents.DocumentExtractionError) as caught:
        documents._user_agent_for(url)
    assert caught.value.reason == "DOCUMENT_SEC_CONTACT_REQUIRED"
    assert "SEC_CONTACT_EMAIL" in (caught.value.detail or "")


@pytest.mark.parametrize(
    ("user_agent", "token"),
    [
        (_DEFAULT_UA, "InvestorAgent/1.0"),
        ("Acme/2.0", "Acme/2.0"),
        ("Mozilla/5.0 (X11) Gecko/2010 Firefox/115.0", "Gecko/2010"),
        ("Mozilla/5.0", "Mozilla/5.0"),
        ("plain text agent", "plain text agent"),
    ],
)
def test_the_product_token_is_the_applications_own(user_agent, token):
    assert documents._product_token(user_agent) == token


def test_the_sec_host_rule_is_data_in_one_place():
    assert documents._CONTACT_REQUIRED_HOSTS == ("sec.gov",)
    assert documents._SEC_HOST in documents._OFFICIAL_HOST_SUFFIXES


@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_sec_contact_is_unset(monkeypatch, empty):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", empty)
    assert Settings(_env_file=None).sec_contact_email is None


def test_the_sec_contact_has_no_default():
    assert Settings.model_fields["sec_contact_email"].default is None


@pytest.mark.parametrize("bad", ["not-an-address", "a b@example.com", "a@b\n.com"])
def test_a_malformed_sec_contact_is_rejected(monkeypatch, bad):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", bad)
    with pytest.raises(ValueError, match="SEC_CONTACT_EMAIL"):
        Settings(_env_file=None)


def test_a_sec_contact_is_trimmed(monkeypatch):
    monkeypatch.setenv("SEC_CONTACT_EMAIL", f"  {_CONTACT} ")
    assert Settings(_env_file=None).sec_contact_email == _CONTACT
