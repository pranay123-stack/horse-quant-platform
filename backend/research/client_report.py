"""The client-facing PDF: ``Horse_Racing_Quant_Strategy_Report.pdf``.

The engineering here is unremarkable. The discipline is the point.

A client report is the document most likely to be read by someone who will not
read the code, will not run the backtest, and may act on it with money. So every
profitability claim in this file is derived from one place — the frozen decision
function's verdict — and there is no code path that can produce an encouraging
summary from a verdict that does not say ``EDGE_CONFIRMED``.

That is enforced by :func:`_executive_summary`, which selects its wording from
the verdict enum rather than from any number, and by a test asserting the phrase
"profitable" cannot appear in a report whose verdict is anything else. ROI is
still shown when it is positive-but-unproven — hiding it would be its own kind
of dishonesty — but it is shown alongside the reason it does not count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.research.protocol import Verdict
from backend.research.validation_report import VERDICT_LABELS, ValidationRun
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

REPORT_FILENAME = "Horse_Racing_Quant_Strategy_Report.pdf"

#: The single point at which this document is allowed to describe the strategy
#: as profitable. Nothing else in the module may make that claim.
PROFITABLE_VERDICT = Verdict.EDGE_CONFIRMED

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#d0d0d0")
GOOD = colors.HexColor("#1b6e3c")
BAD = colors.HexColor("#9b2226")
WARN = colors.HexColor("#8a6100")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=22, leading=27, textColor=INK, spaceAfter=4
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=16
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=14,
            leading=18,
            textColor=INK,
            spaceBefore=16,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=14,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
            spaceAfter=7,
        ),
        "verdict": ParagraphStyle(
            "verdict", parent=base["Heading1"], fontSize=17, leading=21, spaceBefore=6, spaceAfter=8
        ),
    }


@dataclass(slots=True)
class ClientReport:
    """The rendered document and the claim it was permitted to make."""

    path: Path
    verdict_label: str
    claims_profitability: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "verdict": self.verdict_label,
            "claims_profitability": self.claims_profitability,
        }


# ---------------------------------------------------------------------------
# The one guarded decision
# ---------------------------------------------------------------------------
def claims_profitability(run: ValidationRun) -> bool:
    """May this document describe the strategy as profitable?

    One function, one condition, consulted everywhere. A positive ROI is not
    sufficient and is not consulted here — that is the whole reason the decision
    function was frozen before any data existed.
    """
    return run.verdict is not None and run.verdict.verdict is PROFITABLE_VERDICT


def _executive_summary(run: ValidationRun) -> tuple[str, str, colors.Color]:
    """Headline, body and colour — selected by verdict, never by ROI."""
    label = VERDICT_LABELS.get(run.verdict.verdict, "INCONCLUSIVE") if run.verdict else "INCONCLUSIVE"
    metrics = run.primary.metrics if run.primary else None
    roi = f"{metrics.roi:+.2%}" if metrics else "n/a"
    bets = f"{metrics.bets:,}" if metrics else "0"

    if claims_profitability(run):
        return (
            label,
            "The pre-registered decision rule confirms a positive edge. Over the "
            f"out-of-sample window the primary strategy returned {roi} across {bets} "
            "bets, cleared the required significance threshold, and was profitable in "
            "the required majority of years. This conclusion was produced by a rule "
            "fixed in code before any data was downloaded.",
            GOOD,
        )

    if run.verdict and run.verdict.verdict is Verdict.SEGMENT_ONLY:
        return (
            label,
            "No edge was confirmed overall. One or more segments survived the "
            "multiplicity-corrected significance threshold, which is suggestive but "
            "is not a validated strategy: segment findings are hypotheses for a "
            "future study, not conclusions from this one. <b>This report does not "
            "support staking money on the strategy as a whole.</b>",
            WARN,
        )

    if run.verdict and run.verdict.verdict is Verdict.INSUFFICIENT_DATA:
        return (
            label,
            "The study could not reach a conclusion. Too few qualifying bets were "
            f"placed ({bets}) for any result to be distinguishable from chance. "
            "<b>No claim about profitability is made or implied.</b>",
            MUTED,
        )

    return (
        label,
        "No edge was found. The strategy did not meet the pre-registered criteria "
        f"for a confirmed edge over the out-of-sample window (primary ROI {roi} over "
        f"{bets} bets). <b>This report does not support staking money on it.</b> "
        "This is a normal and useful outcome: betting markets are efficient enough "
        "that most apparent edges do not survive honest out-of-sample testing.",
        BAD,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _table(rows: list[list[str]], *, widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _frame_rows(frame: pd.DataFrame, *, limit: int = 12) -> list[list[str]]:
    if frame.empty:
        return [["(no data)"]]
    header = [str(column) for column in frame.columns]
    body = [
        ["" if pd.isna(value) else str(value) for value in record]
        for record in frame.head(limit).itertuples(index=False)
    ]
    return [header, *body]


def _metric_rows(run: ValidationRun) -> list[list[str]]:
    metrics = run.primary.metrics if run.primary else None
    if metrics is None:
        return [["Metric", "Value"], ["(no bets placed)", "-"]]
    return [
        ["Metric", "Value"],
        ["Bets placed", f"{metrics.bets:,}"],
        ["Strike rate", f"{metrics.strike_rate:.2%}"],
        ["Average price taken", f"{metrics.average_odds:.2f}"],
        ["Total staked", f"{metrics.staked:,.0f}"],
        ["Profit", f"{metrics.profit:+,.0f}"],
        ["Return on investment", f"{metrics.roi:+.2%}"],
        ["Profit factor", f"{metrics.profit_factor:.3f}"],
    ]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _build_story(run: ValidationRun, style: dict[str, ParagraphStyle]) -> list[Any]:
    protocol, audit = run.protocol, run.audit
    metrics = run.primary.metrics if run.primary else None
    label, summary, colour = _executive_summary(run)

    story: list[Any] = [
        Paragraph("UK Horse Racing — Quantitative Strategy Validation", style["title"]),
        Paragraph(
            f"Out-of-sample validation under pre-registered protocol "
            f"{protocol.version} (fingerprint {protocol.fingerprint}, registered "
            f"{protocol.registered_at}) &middot; generated {run.generated_at[:19]}",
            style["subtitle"],
        ),
    ]

    if run.notes:
        story.append(
            Paragraph(
                "<b>WARNING — this run did not pass its data-quality gates.</b> " + " ".join(run.notes),
                ParagraphStyle("warn", parent=style["body"], textColor=BAD),
            )
        )

    # --- 1. executive summary ---------------------------------------------
    story += [
        Paragraph("1. Executive summary", style["h1"]),
        Paragraph(label, ParagraphStyle("v", parent=style["verdict"], textColor=colour)),
        Paragraph(summary, style["body"]),
        Paragraph(
            "The conclusion above is produced mechanically by a decision rule that was "
            "written and content-hashed before any data existed. It cannot be influenced "
            "by the results, and no wording in this document may contradict it.",
            style["note"],
        ),
    ]

    # --- 2. data coverage --------------------------------------------------
    story += [
        Paragraph("2. Data coverage", style["h1"]),
        _table(
            [
                ["Measure", "Value"],
                ["Source", str(audit.source)],
                ["Period", f"{audit.date_from} to {audit.date_to}"],
                ["Racing days", f"{audit.racing_days:,}"],
                ["Races", f"{audit.races:,}"],
                ["Runners", f"{audit.runners:,}"],
                ["Runners with a price", f"{audit.odds_coverage:.1%}"],
                ["Races with results", f"{audit.result_coverage:.1%}"],
                ["Out-of-sample rows scored", f"{run.predictions_rows:,}"],
            ],
            widths=[70 * mm, 60 * mm],
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            f"Training used {protocol.train_start} to {protocol.train_end}; validation "
            f"{protocol.valid_start} to {protocol.valid_end}; and the out-of-sample test "
            f"window {protocol.test_start} to {protocol.test_end}, with a "
            f"{protocol.embargo_days}-day embargo between splits. Splits are by date, "
            "never at random: a random split would let the model learn from races that "
            "had not yet been run.",
            style["body"],
        ),
    ]

    # --- 3. model performance ---------------------------------------------
    story += [
        Paragraph("3. Model performance", style["h1"]),
        Paragraph(
            "Race-level log loss against every baseline. Lower is better. The market "
            "row is the only comparison that decides anything — beating a random or "
            "uniform forecast is table stakes for any racing model.",
            style["body"],
        ),
        _table(_frame_rows(run.baselines)),
    ]

    # --- 4. calibration ----------------------------------------------------
    story += [
        PageBreak(),
        Paragraph("4. Probability calibration", style["h1"]),
        Paragraph(
            "Ranking runners correctly is not sufficient. Stakes are computed from the "
            "probability itself, so a horse quoted at a 20% chance that wins 12% of the "
            "time converts a genuine edge into a genuine loss.",
            style["body"],
        ),
        _table(
            [
                ["Measure", "Value"],
                ["Expected calibration error", f"{run.calibration_error:.4f}"],
                ["Bins compared", f"{len(run.calibration.predicted)}"],
            ],
            widths=[70 * mm, 60 * mm],
        ),
        Spacer(1, 4 * mm),
    ]
    if run.calibration.predicted:
        rows = [["Predicted", "Observed", "Runners"]]
        rows += [
            [f"{predicted:.3f}", f"{observed:.3f}", f"{count:,}"]
            for predicted, observed, count in zip(
                run.calibration.predicted,
                run.calibration.observed,
                run.calibration.counts,
                strict=True,
            )
        ]
        story.append(_table(rows, widths=[40 * mm, 40 * mm, 40 * mm]))

    # --- 5. strategy results ----------------------------------------------
    story += [
        Paragraph("5. Betting strategy results", style["h1"]),
        Paragraph(
            f"All strategies replayed over identical predictions at "
            f"{protocol.slippage:.0%} slippage and {protocol.commission:.0%} commission. "
            f"The pre-registered primary hypothesis is "
            f"<b>{protocol.primary_strategy} / {protocol.primary_staking}</b>; every "
            "other row is a secondary test and carries a higher significance bar.",
            style["body"],
        ),
        _table(_metric_rows(run)),
    ]

    # --- 6. risk -----------------------------------------------------------
    story += [Paragraph("6. Risk analysis", style["h1"])]
    if metrics is not None:
        story.append(
            _table(
                [
                    ["Measure", "Value"],
                    ["Starting bankroll", f"{metrics.starting_bankroll:,.0f}"],
                    ["Final bankroll", f"{metrics.final_bankroll:,.0f}"],
                    ["Peak bankroll", f"{metrics.peak_bankroll:,.0f}"],
                    ["Average stake", f"{metrics.average_stake:,.2f}"],
                    ["Longest losing streak", f"{metrics.longest_losing_streak:,} bets"],
                    ["Sharpe per bet", f"{metrics.sharpe_per_bet:.4f}"],
                    ["Bet rate (of races seen)", f"{metrics.bet_rate:.2%}"],
                ],
                widths=[70 * mm, 60 * mm],
            )
        )
        story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Backtested risk is a lower bound on real risk. The simulation does not "
            "model account restriction or closure, which is the normal commercial "
            "response to a consistently winning customer; it assumes the quoted price "
            "is available in the size required; and it does not model a market that "
            "moves against a value bettor specifically, which it does.",
            style["note"],
        )
    )

    # --- 7. drawdown -------------------------------------------------------
    story += [Paragraph("7. Drawdown analysis", style["h1"])]
    if metrics is not None:
        story.append(
            _table(
                [
                    ["Measure", "Value"],
                    ["Maximum drawdown", f"{metrics.max_drawdown:,.0f}"],
                    ["Maximum drawdown (%)", f"{metrics.max_drawdown_pct:.2%}"],
                    ["Longest losing streak", f"{metrics.longest_losing_streak:,} bets"],
                ],
                widths=[70 * mm, 60 * mm],
            )
        )
        story.append(Spacer(1, 4 * mm))
    if run.primary is not None and not run.primary.drawdowns.empty:
        story.append(Paragraph("Deepest drawdown periods", style["h2"]))
        story.append(_table(_frame_rows(run.primary.drawdowns, limit=6)))
    story.append(
        Paragraph(
            "A drawdown observed once in a single out-of-sample window is a sample, "
            "not a bound. The deepest drawdown a strategy will experience is almost "
            "always still ahead of it.",
            style["note"],
        )
    )

    # --- 8. significance ---------------------------------------------------
    story += [
        PageBreak(),
        Paragraph("8. Statistical significance", style["h1"]),
        _table(
            [
                ["Quantity", "Observed", "Required"],
                [
                    "Bets (primary)",
                    f"{metrics.bets:,}" if metrics else "0",
                    f"{protocol.min_bets:,}",
                ],
                [
                    "t-statistic (primary)",
                    f"{metrics.t_statistic:+.2f}" if metrics else "0.00",
                    f"{protocol.min_t_statistic:.2f}",
                ],
                [
                    "Profitable years",
                    f"{run.consistency.profitable_year_fraction:.0%}",
                    f"{protocol.min_profitable_year_fraction:.0%}",
                ],
                [
                    f"Segment bar (Bonferroni, {protocol.secondary_tests} tests)",
                    f"{run.consistency.corrected_threshold:.2f}",
                    "-",
                ],
            ],
            widths=[70 * mm, 40 * mm, 30 * mm],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Does the model know anything the price does not?", style["h2"]),
        Paragraph(run.encompassing.verdict, style["body"]),
        _table(
            [
                ["Term", "Coefficient", "z"],
                [
                    "Market price",
                    f"{run.encompassing.market_coefficient:+.4f}",
                    f"{run.encompassing.market_z:+.2f}",
                ],
                [
                    "Model",
                    f"{run.encompassing.model_coefficient:+.4f}",
                    f"{run.encompassing.model_z:+.2f}",
                ],
            ],
            widths=[70 * mm, 40 * mm, 30 * mm],
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "This is a forecast encompassing regression. Because the bookmakers' own "
            "implied probability is already in the equation, the model's coefficient "
            "measures only what it adds <i>given that the price is already known</i>. "
            "It is a considerably harder test to pass than beating the market on log "
            "loss, and it is the one worth reporting.",
            style["note"],
        ),
    ]

    # --- 9. final verdict --------------------------------------------------
    story += [
        Paragraph("9. Final verdict", style["h1"]),
        KeepTogether(
            [
                Paragraph(label, ParagraphStyle("v2", parent=style["verdict"], textColor=colour)),
                Paragraph(
                    run.verdict.headline if run.verdict else "No verdict was computed.",
                    style["body"],
                ),
            ]
        ),
    ]
    if run.verdict:
        story.append(Paragraph("Reasoning, in the order the frozen rule evaluates it:", style["h2"]))
        for index, reason in enumerate(run.verdict.reasons, start=1):
            story.append(Paragraph(f"{index}. {reason}", style["body"]))

    story.append(
        Paragraph(
            "This document reports a research result. It is not financial advice and "
            "not a recommendation to stake money. Betting markets are efficient enough "
            "that most apparent edges are overfitting, stale prices, or unmodelled "
            "costs; never stake money you cannot afford to lose.",
            style["note"],
        )
    )
    return story


# ---------------------------------------------------------------------------
def generate_client_report(run: ValidationRun, path: str | Path) -> ClientReport:
    """Render the client PDF.

    Returns what was written *and* whether it was permitted to claim
    profitability, so the caller can assert on it rather than parse the PDF.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        title="UK Horse Racing — Quantitative Strategy Validation",
        author="Horse Quant Platform",
        subject=f"Out-of-sample validation under protocol {run.protocol.version}",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(_build_story(run, _styles()))

    label = VERDICT_LABELS.get(run.verdict.verdict, "INCONCLUSIVE") if run.verdict else "INCONCLUSIVE"
    report = ClientReport(path=target, verdict_label=label, claims_profitability=claims_profitability(run))
    logger.info("client report written", extra=report.as_dict())
    return report


__all__ = [
    "PROFITABLE_VERDICT",
    "REPORT_FILENAME",
    "ClientReport",
    "claims_profitability",
    "generate_client_report",
]
