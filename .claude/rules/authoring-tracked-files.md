---
paths:
  - ".claude/**"
  - "docs/**"
  - "*.md"
---

# Authoring a tracked instruction file

Know which zone you are editing before writing any reference into it.

| Zone | Examples | Visibility |
|---|---|---|
| **1 Public** | code, `docs/`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md` | tracked, human-facing |
| **2 Agent layer** | `.claude/rules/`, `.claude/skills/` | tracked, tool-facing |
| **3 Local** | root instruction file, operator notes, local settings, migration ledger | gitignored |
| **4 User-level** | agent configuration outside this repo | not in the repo |

Two orthogonal rules, one hook each.

**Visibility — the test is *obtainable*, not *ignored*.** Gitignored is not by itself
disqualifying; "would a reader end up with this by setting up and running the project?"
is. Three tiers:

- **Nameable.** Produced by normal setup or operation — the environment file created
  from the tracked template, `results/`, capture bundles, runtime state. Name these
  freely, but say how they come to exist.
- **Not nameable.** Ignored and *not* part of operating this repo: another tool's config,
  operator notes, a local archive, anything outside the repo directory, and every
  `.env*` variant other than the template and the one setup creates. Describe these with
  no path.
- **Contents — never, for any tier.** Naming the environment file is fine; quoting a
  line or a value out of it is not, and that holds for everything above.

Absolute and home-relative paths are unobtainable by construction. Run
`git check-ignore -q <path>` when unsure. `tracked-deps` enforces this.

**Kind — human documentation never cites agent metadata.** Zone 1 must not name zone 2,
3 or 4, tracked or not: not the agent directories, not the root instruction file, not
another agent's equivalent. State authority instead — *the implementation and its tests
are authoritative*. `doc-layering` enforces this.

**The rules compose; zone 2 is not exempt from the first.** `doc-layering` skips files
that are themselves agent layer, since agent files may cite zone 1 and each other — but
that skip is about *kind* only. Visibility still binds: a tracked rule or skill may not
name another tool's ignored config directory, a skill that exists only in user-level
configuration, or anything else in zone 3 or 4. Only `tracked-deps` catches those, and
**a bare name with no slash is still a reference.**

Run `make docs-guards` and `make claude-metadata-guards` before handing over a Claude
metadata change; otherwise the hooks fire at commit time, too late when someone else
commits. `make check-all` and CI include both.

Both hooks take `tracked-deps-ok(<path>)` / `doc-layering-ok(<token>)`, but each exempts
that token **throughout the file**, silencing the next unrelated occurrence too. Reword
instead: this repository has zero exemptions and should keep it that way.

## Secrets

Name the environment file as the *reader's* file — where they put their own keys — and
never reproduce anything from it: not a line, not a value, not a paraphrase of one. No
credential, token, account number, or host-specific URL belongs in zone 1 or 2,
examples included.

## Before adding a rule or skill

Ask whether it is really about this repository. Guidance that holds on any codebase —
shell hazards, commit style, planning format, dependency policy — belongs in zone 4,
where every project gets it. **Propose the move and get agreement; do not ship it here
by default.** A rule loads whole on every matching edit, so keep it short and put the
narrative in `docs/`.

Another coding tool's metadata is outside this instruction layer. Do not import it,
use it as guidance, copy or cite its contents, or modify it. If two instruction layers
conflict, describe the conflict generically and leave the other layer untouched.
