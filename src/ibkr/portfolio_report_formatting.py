"""Shared line-formatting primitives for the portfolio text report."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from src.ibkr.models import ReconciliationItem
from src.ibkr.order_presentation import build_live_order_note, find_live_order
from src.ibkr.portfolio_presentation import get_action_label

ACTION_SYMBOLS = {
    "BUY": "BUY",
    "SELL": "SELL",
    "TRIM": "TRIM",
    "ADD": "ADD",
    "HOLD": "HOLD",
    "REVIEW": "REVIEW",
    "REMOVE": "REMOVE",
}
DIVIDER = "═" * 54
DETAIL_INDENT = "             "
DETAIL_WRAP_WIDTH = 96
# Detail lines lead with a fixed-width key label ("verdict:", "scores:", ...)
# so the load-bearing item of every line sits in one scannable column.
DETAIL_LABEL_WIDTH = 11
# Minimum content columns a wrapped listing must retain beside its label.
_MIN_LISTING_ROOM = 20


def currency_prefix(currency: str | None) -> str:
    return f"{currency.upper()} " if currency else "? "


def item_currency(item: ReconciliationItem) -> str | None:
    if item.analysis and item.analysis.currency:
        return item.analysis.currency
    if item.ibkr_position and item.ibkr_position.currency:
        return item.ibkr_position.currency
    return None


def urgency_prefix(item: ReconciliationItem) -> str:
    return {"HIGH": "⚠ ", "MEDIUM": "△ ", "LOW": "  "}.get(item.urgency, "  ")


def bar_chart(pct: float, limit: float, width: int = 14) -> str:
    filled = min(width, round(pct / max(limit, 0.1) * width))
    bar = "█" * filled + "░" * (width - filled)
    warn = " ⚠" if pct >= limit * 0.9 else ""
    return f"{bar}{warn}"


def display_ticker(item: ReconciliationItem) -> str:
    """Exchange-qualified ticker for non-US listings; bare symbol otherwise.

    A bare base symbol is ambiguous across exchanges (SGX AGS vs Brussels AGS;
    Tokyo 6741 vs Taiwan 6741) — two distinct positions must never render
    identically. US listings have no suffix, so the bare symbol stays.
    """
    if item.ticker.has_suffix:
        return item.ticker.yf
    return item.ticker.ibkr


def normalize_reason(reason: str) -> str:
    return reason.replace("DO_NOT_INITIATE", "REJECT").replace(
        "Verdict → ", "Verdict: "
    )


def split_reason(reason: str) -> tuple[str, str | None]:
    """Split a reason into a short head and its explanation for a detail line."""
    head, separator, detail = reason.partition(" — ")
    if separator and detail.strip():
        return head.rstrip(), detail.strip()
    return reason, None


def wrap_listing(
    label: str,
    entries: list[str],
    *,
    separator: str = ", ",
    width: int = DETAIL_WRAP_WIDTH,
) -> list[str]:
    """Wrap ``label`` + comma-joined entries, aligning continuations under it.

    A label that leaves too little room (e.g. a non-canonical exchange key in a
    grouped concentration reason) goes on its own line with the entries wrapped
    under an indent, instead of crashing textwrap with a non-positive width.
    """
    if len(label) + _MIN_LISTING_ROOM >= width:
        indent = " " * (len(label) - len(label.lstrip(" "))) + "    "
        return [
            label.rstrip(),
            *ReportBuffer.wrap_banner_value(
                indent,
                separator.join(entries),
                width=width,
                max_lines=len(entries) + 1,
            ),
        ]
    return ReportBuffer.wrap_banner_value(
        label,
        separator.join(entries),
        width=width,
        max_lines=len(entries) + 1,
    )


def as_of_date(reason: str) -> str:
    normalized = normalize_reason(reason)
    paren = normalized.rfind("(")
    if paren != -1 and normalized.endswith(")"):
        return f"analyzed {normalized[paren + 1 : -1]}"
    return normalized


@dataclass
class ReportBuffer:
    """Mutable output buffer with report-only formatting configuration."""

    lines: list[str]
    show_recommendations: bool
    settled_cash_usd: float
    live_orders: list[dict]

    def section(self, title: str, subtitle: str = "") -> None:
        self.lines.append(DIVIDER)
        header = f"  {title}"
        if subtitle:
            header += f"  ({subtitle})"
        self.lines.extend((header, DIVIDER, ""))

    def append_wrapped_segments(
        self,
        segments: list[str],
        *,
        indent: str = DETAIL_INDENT,
        separator: str = "  ·  ",
        width: int = DETAIL_WRAP_WIDTH,
    ) -> None:
        clean_segments = [
            segment.strip() for segment in segments if segment and segment.strip()
        ]
        if not clean_segments:
            return
        current = indent
        for segment in clean_segments:
            addition = segment if current == indent else f"{separator}{segment}"
            if len(current) + len(addition) <= width:
                current += addition
                continue
            if current != indent:
                self.lines.append(current)
            current = indent + segment
        if current != indent:
            self.lines.append(current)

    @staticmethod
    def wrap_banner_value(
        label: str,
        value: str,
        *,
        width: int,
        max_lines: int = 2,
    ) -> list[str]:
        if not value:
            return []
        subsequent = " " * len(label)
        # Floor the content width: a label near/over `width` must degrade to a
        # narrow wrap, never crash textwrap with a non-positive width.
        content_width = max(width - len(subsequent), _MIN_LISTING_ROOM)
        # ``textwrap.wrap``'s ``width`` is the *total* line width including the
        # indent, so pass the full target width (floored so ``width - indent``
        # stays >= _MIN_LISTING_ROOM); ``content_width`` alone would leave
        # ``width - 2*len(indent)`` of room and break every character on a long
        # label (the "Information Technology" bucket regression).
        wrapped = textwrap.wrap(
            value,
            width=content_width + len(subsequent),
            initial_indent=label,
            subsequent_indent=subsequent,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if len(wrapped) <= max_lines:
            return wrapped
        retained = wrapped[: max_lines - 1]
        remaining = " ".join(line.strip() for line in wrapped[max_lines - 1 :])
        retained.append(
            subsequent
            + textwrap.shorten(
                remaining,
                width=content_width,
                placeholder=" [truncated]",
            )
        )
        return retained

    def order_line(self, item: ReconciliationItem, currency: str | None) -> str:
        parts = [
            f"{urgency_prefix(item)}{ACTION_SYMBOLS.get(item.action, item.action):<6}  "
            f"{display_ticker(item):<12}"
        ]
        if self.show_recommendations and item.action != "REVIEW":
            if item.suggested_quantity:
                parts.append(f"  {abs(item.suggested_quantity)} shares")
            if item.suggested_price:
                parts.append(
                    f"  @ {currency_prefix(currency)}{item.suggested_price:,.2f}"
                )
            if item.suggested_order_type:
                parts.append(f"  {item.suggested_order_type}")
            if item.action == "BUY" and not item.suggested_quantity:
                parts.append("  (quantity unavailable — inspect before placing order)")
            if not item.suggested_price:
                parts.append("  (no entry price — re-run analysis)")
        return "".join(parts)

    @staticmethod
    def holding_line(item: ReconciliationItem, currency: str | None) -> str | None:
        pos = item.ibkr_position
        if pos is None or not pos.quantity:
            return None
        parts = [f"{abs(pos.quantity):,.0f} shares"]
        if pos.current_price_local:
            parts.append(
                f"last {currency_prefix(currency)}{pos.current_price_local:,.2f}"
            )
        if pos.market_value_usd:
            parts.append(f"position value ~${pos.market_value_usd:,.0f} USD")
        return f"{DETAIL_INDENT}{'holding:':<{DETAIL_LABEL_WIDTH}}" + "  ·  ".join(
            parts
        )

    def proceeds_line(self, item: ReconciliationItem) -> str | None:
        if not self.show_recommendations or not item.cash_impact_usd:
            return None
        parts = [f"~${item.cash_impact_usd:,.0f} USD"]
        if item.settlement_date:
            parts.append(f"spendable on {item.settlement_date}")
        return f"{DETAIL_INDENT}{'proceeds:':<{DETAIL_LABEL_WIDTH}}" + "  ·  ".join(
            parts
        )

    def cost_line(self, item: ReconciliationItem, label: str = "Cost") -> str | None:
        if not self.show_recommendations or not item.cash_impact_usd:
            return None
        funded = (
            f"use already-settled cash (${self.settled_cash_usd:,.0f} available)"
            if self.settled_cash_usd > 0
            else "use already-settled cash"
        )
        return (
            f"{DETAIL_INDENT}{label}: ~${abs(item.cash_impact_usd):,.0f} USD"
            f"  ·  {funded}"
        )

    def display_data_line(self, item: ReconciliationItem) -> None:
        analysis = item.analysis
        pos = item.ibkr_position
        if not analysis:
            return
        symbol = currency_prefix(item_currency(item))
        health = (
            f"{analysis.health_adj:.0f}" if analysis.health_adj is not None else "?"
        )
        growth = (
            f"{analysis.growth_adj:.0f}" if analysis.growth_adj is not None else "?"
        )
        parts = [f"Health:{health}%  Growth:{growth}%"]
        if analysis.zone:
            parts.append(f"Risk:{analysis.zone}")
        if (
            analysis.entry_price
            and pos
            and pos.current_price_local
            and analysis.entry_price > 0
        ):
            change = (
                (pos.current_price_local - analysis.entry_price)
                / analysis.entry_price
                * 100
            )
            suffix = (
                "  (⚠ unit mismatch?)" if abs(change) >= 90.0 else f"  ({change:+.1f}%)"
            )
            parts.append(
                f"analysis entry {symbol}{analysis.entry_price:,.2f}  "
                f"now {symbol}{pos.current_price_local:,.2f}{suffix}"
            )
        self.append_wrapped_segments(parts, separator="  |  ")

    @staticmethod
    def score_line(item: ReconciliationItem) -> str | None:
        analysis = item.analysis
        if not analysis or (
            analysis.health_adj is None and analysis.growth_adj is None
        ):
            return None
        parts: list[str] = []
        if analysis.analysis_date and analysis.age_days < 9999:
            parts.append(analysis.analysis_date)
        if analysis.health_adj is not None and analysis.growth_adj is not None:
            parts.append(
                f"Health:{analysis.health_adj:.0f}  Growth:{analysis.growth_adj:.0f}"
            )
        elif analysis.health_adj is not None:
            parts.append(f"Health:{analysis.health_adj:.0f}")
        else:
            parts.append(f"Growth:{analysis.growth_adj:.0f}")
        if analysis.zone:
            parts.append(f"Risk zone:{analysis.zone}")
        verdict = analysis.verdict or ""
        if analysis.conviction:
            verdict += f" ({analysis.conviction})" if verdict else analysis.conviction
        if verdict:
            parts.append(verdict)
        return f"{DETAIL_INDENT}{'analysis:':<{DETAIL_LABEL_WIDTH}}" + "  ·  ".join(
            parts
        )

    @staticmethod
    def pnl_line(item: ReconciliationItem) -> str | None:
        from src.ibkr.portfolio_presentation import cost_basis_unit_mismatch

        pos = item.ibkr_position
        if (
            not pos
            or pos.avg_cost_local <= 0
            or pos.current_price_local <= 0
            or pos.quantity == 0
        ):
            return None
        pct = (pos.current_price_local - pos.avg_cost_local) / pos.avg_cost_local * 100
        if cost_basis_unit_mismatch(pos):
            return (
                f"{DETAIL_INDENT}{'est. P&L:':<{DETAIL_LABEL_WIDTH}}"
                "(⚠ cost basis may have currency-unit mismatch)"
            )
        sell_qty = abs(item.suggested_quantity or pos.quantity)
        pnl_local = (pos.current_price_local - pos.avg_cost_local) * sell_qty
        symbol = currency_prefix(pos.currency)
        sign = "+" if pnl_local >= 0 else "-"
        label = "est. gain:" if pnl_local >= 0 else "est. loss:"
        tax_note = "  ·  verify holding period in IBKR" if pnl_local > 0 else ""
        return (
            f"{DETAIL_INDENT}{label:<{DETAIL_LABEL_WIDTH}}{sign}{symbol}"
            f"{abs(pnl_local):,.0f}"
            f"  ({pct:+.1f}% vs IBKR cost basis {symbol}{pos.avg_cost_local:,.2f})"
            f"{tax_note}"
        )

    @staticmethod
    def soft_rejection_score_segments(item: ReconciliationItem) -> list[str]:
        analysis = item.analysis
        if not analysis:
            return []
        segments: list[str] = []
        if analysis.health_adj is not None:
            segments.append(f"H:{analysis.health_adj:.0f}%")
        if analysis.growth_adj is not None:
            segments.append(f"G:{analysis.growth_adj:.0f}%")
        if analysis.zone:
            segments.append(f"Risk:{analysis.zone}")
        return segments

    @staticmethod
    def soft_rejection_thesis_segment(item: ReconciliationItem) -> str | None:
        analysis = item.analysis
        pos = item.ibkr_position
        if (
            not analysis
            or not analysis.entry_price
            or not pos
            or not pos.current_price_local
        ):
            return None
        symbol = currency_prefix(item_currency(item))
        change = (
            (pos.current_price_local - analysis.entry_price)
            / analysis.entry_price
            * 100
        )
        suffix = " (unit mismatch?)" if abs(change) >= 90.0 else f" ({change:+.1f}%)"
        return (
            f"thesis: entry {symbol}{analysis.entry_price:,.2f} -> "
            f"now {symbol}{pos.current_price_local:,.2f}{suffix}"
        )

    def soft_rejection_pnl_segments(self, item: ReconciliationItem) -> list[str]:
        from src.ibkr.portfolio_presentation import cost_basis_unit_mismatch

        pos = item.ibkr_position
        if (
            not pos
            or pos.avg_cost_local <= 0
            or pos.current_price_local <= 0
            or pos.quantity == 0
        ):
            return []
        pct = (pos.current_price_local - pos.avg_cost_local) / pos.avg_cost_local * 100
        if cost_basis_unit_mismatch(pos):
            return ["P/L: cost basis may have currency-unit mismatch"]
        sell_qty = abs(item.suggested_quantity or pos.quantity)
        pnl_local = (pos.current_price_local - pos.avg_cost_local) * sell_qty
        symbol = currency_prefix(pos.currency)
        sign = "+" if pnl_local >= 0 else "-"
        segments = [
            f"P/L vs IBKR: {sign}{symbol}{abs(pnl_local):,.0f} "
            f"({pct:+.1f}% vs {symbol}{pos.avg_cost_local:,.2f})"
        ]
        if pnl_local > 0:
            segments.append("holding period: verify in IBKR")
        if self.show_recommendations and item.cash_impact_usd:
            segments.append(f"proceeds ~${item.cash_impact_usd:,.0f} USD")
        if self.show_recommendations and item.settlement_date:
            segments.append(f"settles {item.settlement_date}")
        return segments

    def labeled_detail(self, label: str, text: str) -> None:
        """A detail line leading with an aligned key label, wrapped under it."""
        prefix = f"{DETAIL_INDENT}{label:<{DETAIL_LABEL_WIDTH}}"
        # wrap_banner_value's ``width`` is the total line width; run to ~110 cols.
        self.lines.extend(
            self.wrap_banner_value(
                prefix, text, width=DETAIL_WRAP_WIDTH + 14, max_lines=3
            )
        )

    def append_reason_detail(self, reason: str) -> None:
        """Reason text on its own labeled detail line(s), never on the header."""
        label, _, rest = reason.partition(": ")
        if label == "Verdict" and rest:
            self.labeled_detail("verdict:", rest)
        else:
            self.labeled_detail("reason:", reason)

    def append_soft_rejection_details(self, item: ReconciliationItem) -> None:
        """One labeled line per detail group (scores, thesis, P/L) —
        deterministic layout instead of greedy packing, which merged groups
        only when they happened to fit."""
        scores = self.soft_rejection_score_segments(item)
        if scores:
            self.labeled_detail("scores:", "  ·  ".join(scores))
        thesis = self.soft_rejection_thesis_segment(item)
        if thesis:
            self.labeled_detail("thesis:", thesis.removeprefix("thesis: "))
        pnl = self.soft_rejection_pnl_segments(item)
        if pnl:
            first = pnl[0].removeprefix("P/L vs IBKR: ").removeprefix("P/L: ")
            self.labeled_detail("P/L:", "  ·  ".join([first, *pnl[1:]]))

    @staticmethod
    def sell_type_label(item: ReconciliationItem) -> str:
        return get_action_label(item)

    def append_thesis_break_line(self, item: ReconciliationItem) -> None:
        """Bear thesis-break triggers — the fundamental exit conditions that
        carry sell authority, shown ahead of legacy downside-price context."""
        analysis = item.analysis
        criteria = getattr(analysis, "kill_criteria", ()) if analysis else ()
        if criteria:
            self.labeled_detail("break if:", "  ·  ".join(criteria))

    def append_fx_split_line(self, item: ReconciliationItem) -> None:
        """Local-price vs implied FX/basis decomposition for non-USD positions.

        The local leg is the thesis-relevant number for an investor who treats
        EM-FX erosion as expected cost; the USD leg stays as quiet NAV context.
        """
        from src.ibkr.portfolio_presentation import fx_return_split_diagnostic

        split, issue = fx_return_split_diagnostic(item.ibkr_position)
        if issue:
            self.labeled_detail("return:", issue)
            return
        if split is None:
            return
        local_pct, fx_pct, usd_pct = split
        self.labeled_detail(
            "return:",
            f"local-price {local_pct:+.1f}%  ·  implied FX/basis {fx_pct:+.1f}%  ·  "
            f"USD {usd_pct:+.1f}%",
        )

    def profit_take_segments(self, item: ReconciliationItem) -> list[str]:
        segments: list[str] = []
        if item.cost_basis_return_pct is not None:
            segments.append(f"gain vs cost: {item.cost_basis_return_pct:+.1f}%")
        if item.ibkr_position:
            segments.append(f"tax: {item.ibkr_position.tax_term}")
        if item.profit_take_reasons:
            segments.append(f"drivers: {item.reason}")
        if self.show_recommendations and item.cash_impact_usd:
            segments.append(f"proceeds ~${item.cash_impact_usd:,.0f} USD")
        if self.show_recommendations and item.settlement_date:
            segments.append(f"settles {item.settlement_date}")
        return segments

    def append_sale_tax_note(self, item: ReconciliationItem) -> None:
        """Add honest tax-lot context to an actual sale recommendation."""
        if item.action != "SELL" or item.ibkr_position is None:
            return
        tax_term = item.ibkr_position.tax_term
        if tax_term == "SHORT_TERM":
            text = "short-term holding indicated; verify selected lots and tax impact in IBKR"
        elif tax_term == "LONG_TERM":
            text = "long-term holding indicated; verify selected lots and realized gain in IBKR"
        else:
            text = "tax-lot holding periods unavailable; verify lots and realized gain in IBKR"
        self.labeled_detail("tax:", text)

    def append_pnl_proceeds(
        self, item: ReconciliationItem, _currency: str | None
    ) -> None:
        # Deliberately separate lines: the joined form routinely exceeded a
        # 120-column terminal.
        pnl = self.pnl_line(item)
        proceeds = self.proceeds_line(item)
        if pnl:
            self.lines.append(pnl)
        if proceeds:
            self.lines.append(proceeds)

    @staticmethod
    def buy_pos_tag(item: ReconciliationItem) -> str:
        if item.action == "ADD":
            pos = item.ibkr_position
            quantity = f"{pos.quantity:,.0f} sh" if pos and pos.quantity else "held"
            return f"[up position — {quantity}]"
        if item.is_watchlist:
            return "[watchlist — new position]"
        return "[untracked — new position]"

    def find_live_order(self, item: ReconciliationItem) -> tuple[dict, str] | None:
        match = find_live_order(item, self.live_orders)
        return (match.order, match.side) if match is not None else None

    def order_note(self, item: ReconciliationItem) -> str | None:
        note = build_live_order_note(item, self.live_orders)
        return f"{DETAIL_INDENT}{note}" if note else None
