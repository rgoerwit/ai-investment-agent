---
paths:
  - "src/memory.py"
  - "src/retrospective.py"
  - "src/embeddings.py"
---

# Memory and the retrospective

## Ticker isolation is the point

Every analysis gets **its own per-ticker collections**. Build them through the
ticker-scoped factory; never reach for a global memory object. Without isolation one
company's data contaminates another's analysis — a chip-shortage document surfacing
inside a bank's write-up — and the contamination is invisible in the output.

Retrieval is filtered by ticker metadata as well as scoped by collection. Collection
names carry an embedding fingerprint, so changing the embedding model cannot silently
return neighbours computed under the old one.

A cleanup helper called **without** a ticker wipes every collection. Production call
sites must always pass one.

## Embedding calls are hard-bounded

The client library treats an unset per-request timeout as *no timeout*, so an
unreachable endpoint blocks forever rather than failing. The constructor health check
runs on a daemon thread with a join timeout, and every embedding call is wrapped in
the hard-timeout helper. Do not "fix" this through the library's own request-options
field, which is declared and never consumed.

## The retrospective may not adjudicate the thesis

Excess return is price minus benchmark. That says only what the country index does not
explain — never *what did*. So:

- **Residual dominance does not establish that the analysis was wrong.** It can be an
  earnings surprise, a takeover rumour, sector rotation, or a data error, and price
  cannot discriminate among them.
- A **missing benchmark makes the outcome unassessable**, not flat. It produces no
  trigger, no model call, and no stored lesson. Never let it collapse into zero, which
  would read a market-wide decline as a company collapse.
- **Could-not-attribute is not market-driven.** Keep the two distinct, because the
  scope stamp gates the prompt rule that tells the model the residual is unexplained.
- A lesson is injectable only when it is attributable **and** carries the metadata
  retrieval needs to apply it. Everything else is retained for review and never
  injected — an unexplained outcome phrased as an imperative still acts as one.

Whether a pre-registered thesis-break trigger actually fired is a question about
filings published after the decision, which this path never fetches. Leave it
unevaluated rather than guessing.

## Provenance versus weighting

Record model names — they are what make runs comparable over time. Do **not** key
judgements on them: a name-to-quality table rots silently, and a lookup that matches
nothing pins its factor at a default while looking like it works. Key on the binding
layer's own closed enum instead, so a new model generation needs no edit.

Full history: `docs/RETROSPECTIVE.md`.
