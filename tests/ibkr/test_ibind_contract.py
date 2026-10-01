"""Contract tests against the *installed* ibind package.

Every other IBKR test mocks the ibind client wholesale, so a release that renames
a method or keyword would leave them green while the real client raises
``TypeError`` at the first live call — on the only layer with sale authority.
These tests bind each call ``src/ibkr/client.py`` makes against ibind's real
signatures instead, so an ibind bump fails here rather than in production.
"""

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

ibind = pytest.importorskip("ibind")

from ibind import IbkrClient  # noqa: E402
from ibind.client.ibkr_utils import OrderRequest, QuestionType  # noqa: E402
from ibind.oauth.oauth1a import OAuth1aConfig  # noqa: E402

CLIENT_SRC = Path(__file__).resolve().parents[2] / "src" / "ibkr" / "client.py"

# method -> (args, kwargs) exactly as src/ibkr/client.py passes them.
IBIND_CALLS: dict[str, tuple[tuple, dict]] = {
    "close": ((), {}),
    "oauth_shutdown": ((), {}),
    "portfolio_accounts": ((), {}),
    "positions": ((), {"account_id": "U123"}),
    "get_ledger": ((), {"account_id": "U123"}),
    "stock_conid_by_symbol": (("7203",), {"default_filtering": False}),
    "initialize_brokerage_session": ((), {"compete": True}),
    "authentication_status": ((), {"log": False}),
    "get_all_watchlists": ((), {"sc": "USER_WATCHLIST"}),
    "get_watchlist_information": (("wl1",), {}),
    "live_orders": ((), {"account_id": "U123", "force": True}),
    "contract_information_by_conid": (("123",), {}),
    "security_definition_by_conid": ((["123"],), {}),
    "place_order": (
        (),
        {"order_request": object(), "answers": {}, "account_id": "U123"},
    ),
}


def _methods_used_by_client() -> set[str]:
    """Every ``self._ibind_client.<name>`` attribute referenced in client.py."""
    tree = ast.parse(CLIENT_SRC.read_text())
    used = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_ibind_client"
        ):
            used.add(node.attr)
    return used


def test_contract_table_covers_every_ibind_call_in_client():
    """A new ibind call in client.py must get a contract entry here."""
    missing = _methods_used_by_client() - set(IBIND_CALLS)
    assert not missing, f"add these ibind calls to IBIND_CALLS: {sorted(missing)}"


@pytest.mark.parametrize("method", sorted(IBIND_CALLS))
def test_ibind_method_accepts_our_arguments(method):
    fn = getattr(IbkrClient, method, None)
    assert callable(fn), f"ibind.IbkrClient no longer has {method}()"
    args, kwargs = IBIND_CALLS[method]
    # Raises TypeError on a renamed/removed parameter or an arity change.
    inspect.signature(fn).bind(None, *args, **kwargs)


def test_ibind_client_constructor_accepts_our_arguments():
    inspect.signature(IbkrClient.__init__).bind(
        None, account_id="U123", use_oauth=True, oauth_config=object()
    )


def test_oauth_config_accepts_every_field_we_set():
    fields = {f.name for f in dataclasses.fields(OAuth1aConfig)}
    ours = {
        "access_token",
        "access_token_secret",
        "consumer_key",
        "encryption_key_fp",
        "signature_key_fp",
        "init_oauth",
        "init_brokerage_session",
        "maintain_oauth",
        "shutdown_oauth",
        "dh_prime",
    }
    assert ours <= fields, f"OAuth1aConfig dropped: {sorted(ours - fields)}"


def test_oauth_config_constructs_with_our_kwargs():
    """Construction must not do I/O or reject our shape (values are dummies)."""
    cfg = OAuth1aConfig(
        access_token="tok",
        access_token_secret="secret",
        consumer_key="CONSUMER1",
        encryption_key_fp="/nonexistent/enc.pem",
        signature_key_fp="/nonexistent/sig.pem",
        init_oauth=True,
        init_brokerage_session=False,
        maintain_oauth=False,
        shutdown_oauth=False,
        dh_prime="ff",
    )
    assert cfg.maintain_oauth is False
    assert cfg.shutdown_oauth is False


def test_order_request_accepts_our_fields():
    req = OrderRequest(
        conid=123,
        side="BUY",
        quantity=10,
        order_type="LMT",
        acct_id="U123",
        price=1.5,
        tif="GTC",
    )
    assert req.conid == 123
    assert req.side == "BUY"


def test_order_request_rejects_unknown_field():
    with pytest.raises(TypeError):
        OrderRequest(
            conid=1, side="BUY", quantity=1, order_type="LMT", acct_id="U1", bogus=1
        )


def test_question_type_is_iterable_enum_for_auto_answers():
    """client.place_order auto-confirms with dict.fromkeys(QuestionType, True)."""
    answers = dict.fromkeys(QuestionType, True)
    assert answers, (
        "QuestionType has no members; auto-confirmation would answer nothing"
    )
    assert all(v is True for v in answers.values())
