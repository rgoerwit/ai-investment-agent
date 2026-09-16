---
paths:
  - "src/ibkr/**"
  - "scripts/portfolio_manager.py"
  - "src/web/ibkr_dashboard/**"
---

# Portfolio reconciliation

This layer is the **only** one with authority to sell, and that authority is a
positive whitelist rather than an absence of objections.

## A portfolio action requires portfolio evidence

A stock-level rejection may trigger a refresh, a review, replacement analysis, or an
exit — but it must not choose among those on its own. `classify_disposition()` returns
the action, its basis, and whether it is executable.

- **Gate scores intact (health and growth ≥ 50) never produce a verdict-driven SELL**,
  and that check runs *before* confirmation. A price drop with intact fundamentals is
  a review: watch fundamentals, not price.
- **Confirmed thesis failure** requires provably full-mode analyses on *both* sides,
  spaced apart. `is_quick_mode` is tri-state, and **unknown carries no sell
  authority** — a legacy artifact proves nothing.
- **Stop and target breaches are review-only.** They carry no independent sale
  authority at all; the reason text must couple the price move to the last health and
  growth scores.
- Compliance flags are exempt from the de-minimis floor: the paperwork costs the same
  on a small position.

`retail_safe_action()` is applied at **every** presentation entry point and downgrades
any sell that is not on the whitelist — including one loaded from a legacy cached
bundle.

## Paying for an analysis requires that one could change the answer

The refresh scheduler spends real money per queued ticker, so eligibility is a
claim about *repairability*, not about severity.

- **`blocks_buy` is two different claims.** A settled gate failure (minted
  `action=AUTO_REJECT`, e.g. `LIQUIDITY_HARD_FAIL`) cannot move during the current
  observation period, so it must not enter the urgent stream; it rejoins the ordinary
  fair cycle near analysis expiry. An evidence gap (`action=REVIEW`) might close with
  more research. Read
  `evidence.indeterminate_flag_types` for scheduling, never the
  `buy_blocking_flag_types` union — that union is for "is a buy blocked", which
  both kinds answer yes to.
- **Never key scheduler state on model or search output.** Buy-blocking flag
  *composition* varies run to run for an unchanged position, so a key built from
  it never repeats and the backoff it guards never fires. Key on the action
  basis — a closed enum — or on nothing.
- **Price never enters scheduler identity.** A review-level breach raises
  urgency; it is not evidence that new research exists to buy.
- **`blocking` policy bounds ordering, not throughput.** Serve urgent first,
  then fill the remaining budget from the normal cycle. Passing only the urgent
  stream idles every slot urgent does not fill.

Full history: `docs/CODEBASE_MEMORY.md`.

## Identity gates executability

An order needs a verified listing mapping: a real contract id, an exact ticker match,
and currency agreement. Matching **fails closed on missing currency** — losing an
analysis match is safer than attaching another issuer's research. Display is
exchange-qualified, so two listings of the same symbol never render identically.

Raw broker tokens cross the market-data boundary only through the canonical identity
resolver. An `IBCID<conid>` token is an instruction to recover identity from validated
cache or contract metadata, never a ticker or yfinance input. If recovery fails, keep
the holding in accounting without constructing a `Ticker`; unresolved inventory
blocks every BUY and ADD, while reviews and reductions on resolved holdings remain
available. Do not represent missing identity as a sector or exchange bucket.

## Money has a unit, and the unit is the currency code

Never discard a currency code you were given, never up-case it (that silently folds a
minor-unit code onto its major unit), and never rescale by venue. Two independent
checks, and neither subsumes the other: **same economy?** compares codes; **same
scale?** compares derived price levels against the reference they came from. Agreement
on the code does not imply agreement on the scale.

Refuse, never repair. An incoherent level is nulled and recorded, not rescaled — these
values gate portfolio actions, and a plausible reconstruction is still a guess about
which side is wrong.

## Zero shares is three states

A row with no shares is *closed* (skip it), *contradictory* if it still reports
material value (data-quality review, in every currency), or an ordinary holding whose
price the broker merely omitted. Resolve all three before the FX guard. A negative
quantity is a short and routes to review — never to the sell branch, whose absolute
value would propose selling more of an already-short position.

## Sessions are pooled

One connection per process, reused, closed exactly once at teardown through a real
server-side logout. Never connect per call: that orphans a server-side session which
lingers until timeout, and concurrent ones compete for the single brokerage-session
slot.

Full history: `docs/CURRENCY.md`.
