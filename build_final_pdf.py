"""Create a self-contained, print-ready FD001 RUL final report PDF."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "Final_FD001_RUL_Implementation_Report.pdf"
FIGURE = REPORTS / "figures" / "ablation_comparison.png"

NAVY = colors.HexColor("#17365D")
PALE = colors.HexColor("#F4F7FA")
GRID = colors.HexColor("#D9D9D9")
GREY = colors.HexColor("#595959")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(doc.leftMargin, 0.82 * cm, "FD001 RUL Implementation Report")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.82 * cm, f"Page {doc.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=26, textColor=colors.black, spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11.5,
            leading=15, textColor=GREY, spaceAfter=3,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["Normal"], fontName="Helvetica", fontSize=8.7,
            leading=11, textColor=GREY, spaceAfter=13,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14,
            leading=17, textColor=colors.black, spaceBefore=10, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.4,
            leading=12.6, textColor=colors.black, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.1,
            leading=11.8, leftIndent=13, firstLineIndent=-8, bulletIndent=2, spaceAfter=2.5,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8,
            leading=10, alignment=TA_CENTER, textColor=GREY, spaceAfter=5,
        ),
        "ref": ParagraphStyle(
            "Ref", parent=base["Normal"], fontName="Helvetica", fontSize=8.2,
            leading=10.5, textColor=GREY, spaceAfter=3,
        ),
    }


def p(text: str, style):
    return Paragraph(text, style)


def table(headers, rows, widths):
    data = [[p(f"<b>{value}</b>", ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8, leading=9.5, textColor=colors.white, alignment=TA_CENTER)) for value in headers]]
    cell_style = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.1, leading=10.1)
    numeric_style = ParagraphStyle("num", parent=cell_style, alignment=TA_CENTER)
    for row in rows:
        formatted = []
        for index, value in enumerate(row):
            formatted.append(p(value, numeric_style if index else cell_style))
        data.append(formatted)
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    result.setStyle(TableStyle(style_commands))
    return result


def bullets(items, style):
    return [Paragraph(item, style, bulletText="•") for item in items]


def main():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=1.65 * cm, rightMargin=1.65 * cm,
        topMargin=1.55 * cm, bottomMargin=1.7 * cm, title="NASA C MAPSS FD001 Remaining Useful Life",
        author="FD001 RUL project", subject="Final implementation report",
    )
    story = []

    story.extend([
        p("NASA C MAPSS FD001 Remaining Useful Life", s["title"]),
        p("Final implementation, model comparison, and deployment assessment", s["subtitle"]),
        p(f"Prepared {date.today().strftime('%d %B %Y')} | CPU-bounded local study", s["meta"]),
        p("Executive summary", s["h1"]),
        p(
            "The selected production candidate is a Histogram Gradient Boosting regressor trained with RUL labels capped at 125 cycles and causal 10- and 30-cycle features. It was selected on held-out engine prefixes and retrained on all FD001 training engines. Its single final official-test evaluation achieved RMSE 13.532, MAE 9.799, R2 0.894, and PHM08 mean 2.852.",
            s["body"],
        ),
        p(
            "The project also implements compact CNN, LSTM, CNN-LSTM, TinyML-oriented, autoencoder, hybrid, random forest, ridge, and mean-baseline paths. The compact CNN-LSTM was the best neural candidate, but it did not outperform the selected tree model. The autoencoder is retained as an unlabeled early-life anomaly signal for triage rather than as a supervised RUL predictor.",
            s["body"],
        ),
        table(
            ["Stage", "RMSE", "MAE", "R2", "PHM08 mean"],
            [
                ["Validation selection split", "12.773", "8.609", "0.877", "2.832"],
                ["Three-split validation mean plus or minus SD", "13.623 ± 1.724", "8.802 ± 0.561", "0.866 ± 0.025", "3.471 ± 1.411"],
                ["Official FD001 test, one final evaluation", "13.532", "9.799", "0.894", "2.852"],
            ],
            [6.7 * cm, 2.3 * cm, 2.3 * cm, 1.8 * cm, 2.6 * cm],
        ),
        Spacer(1, 7),
        p("1  Problem and data", s["h1"]),
        p(
            "The task predicts remaining useful life from multivariate turbofan telemetry. FD001 records a unit identifier, cycle, three operating settings, and 21 sensor channels. Training trajectories run to failure; test trajectories are right-censored and have one supplied RUL target for each engine endpoint.",
            s["body"],
        ),
        table(
            ["Split", "Rows", "Engines", "Observed life or cycles", "Audit result"],
            [
                ["Training", "20,631", "100", "128 to 362 cycles", "All quality checks passed"],
                ["Test", "13,096", "100", "31 to 303 observed cycles", "All quality checks passed"],
                ["Test RUL vector", "100", "100", "7 to 145 cycles", "Aligned to sorted engine IDs"],
            ],
            [2.6 * cm, 1.8 * cm, 1.7 * cm, 4.2 * cm, 5.9 * cm],
        ),
        Spacer(1, 5),
        p("No missing or non-finite values, duplicate unit-cycle rows, cycle gaps, invalid IDs, or ordering faults were found. Seven near-constant sensors were screened in an ablation, but removing them did not improve the selected result.", s["body"]),
        p("2  Evaluation protocol and safeguards", s["h1"]),
    ])
    story.extend(bullets([
        "Engines, rather than individual rows, are split into 80 development and 20 validation engines.",
        "Validation evaluates 50%, 70%, and 90% prefixes of held-out trajectories, creating 60 test-like endpoint cases without exposing future cycles as features.",
        "Features are causal and reset at every engine boundary. Difference-to-initial-reading features use only the first observed point for that engine.",
        "Model selection ran without opening test_FD001.txt or RUL_FD001.txt. The selected model was then scored once on official endpoints.",
        "PHM08 is reported alongside RMSE, MAE, R2, bias, and error-percentile measures. Its asymmetric penalty is stricter for RUL overprediction.",
    ], s["bullet"]))

    story.append(PageBreak())
    story.extend([
        p("3  Models implemented", s["h1"]),
        table(
            ["Family", "Role in study", "Result or status"],
            [
                ["Histogram Gradient Boosting", "Selected tabular RUL model", "Best clean validation and final official score"],
                ["CNN, LSTM, CNN-LSTM", "Compact sequence candidates", "CNN-LSTM best neural candidate"],
                ["TinyML-oriented neural regressor", "Small footprint deployment comparison", "Historical official-test comparison"],
                ["Autoencoder", "Early-life reconstruction anomaly signal", "Used for anomaly persistence and triage"],
                ["Hybrid TinyML plus autoencoder", "Joint RUL and anomaly comparison", "Historical deployment comparison"],
                ["Random forest, ridge, mean", "Classical baselines", "Retained for benchmark context"],
            ],
            [4.3 * cm, 6.2 * cm, 6.0 * cm],
        ),
        Spacer(1, 5),
        p("The compact sequence study uses CPU-bounded settings: four threads, batch size 256, hidden width 16, a 20-epoch maximum, and validation early stopping. This kept training suitable for a 16 GB RAM laptop with integrated graphics.", s["body"]),
        p("4  Ablation and model comparison", s["h1"]),
        p("Label capping changed the most important behavior: validation PHM08 mean fell from 19.728 for the uncapped 5/20 control to 6.286 with a 125-cycle cap. The selected 10/30 feature variant reduced it again to 2.832.", s["body"]),
        table(
            ["Candidate", "Validation RMSE", "Validation MAE", "R2", "PHM08 mean"],
            [
                ["HGB uncapped control, 5/20 features", "18.338", "12.409", "0.747", "19.728"],
                ["HGB cap 125, 5/20 features", "16.417", "11.510", "0.797", "6.286"],
                ["HGB cap 125, 10/30 features - selected", "12.773", "8.609", "0.877", "2.832"],
                ["Compact CNN-LSTM", "16.199", "12.488", "0.803", "4.010"],
                ["Compact CNN", "17.525", "13.251", "0.769", "5.428"],
                ["Compact LSTM", "18.592", "13.844", "0.740", "9.125"],
            ],
            [7.0 * cm, 2.8 * cm, 2.8 * cm, 1.6 * cm, 2.7 * cm],
        ),
        Spacer(1, 6),
    ])
    if FIGURE.exists():
        image = Image(str(FIGURE), width=15.4 * cm, height=8.55 * cm)
        story.extend([image, p("Figure 1. Validation ablation comparison. Lower values are better.", s["caption"])])

    story.append(PageBreak())
    story.extend([
        p("5  Official test result and error profile", s["h1"]),
        p("After selection, the HGB model was retrained using all 100 FD001 training engines. It was evaluated once against the 100 official test endpoints. The near-zero bias and balanced direction counts indicate no material systematic over- or under-estimation in this run.", s["body"]),
        table(
            ["Metric", "Value", "Metric", "Value"],
            [
                ["RMSE", "13.532", "MAE", "9.799"],
                ["R2", "0.894", "PHM08 sum", "285.222"],
                ["PHM08 mean", "2.852", "Bias, prediction minus actual", "0.157 cycles"],
                ["P90 absolute error", "24.135 cycles", "Maximum absolute error", "38.845 cycles"],
                ["Over / under predictions", "50 / 50", "Official endpoint count", "100"],
            ],
            [4.4 * cm, 3.4 * cm, 5.0 * cm, 3.2 * cm],
        ),
        Spacer(1, 6),
        p("6  On-device anomaly and maintenance workflow", s["h1"]),
        p("An early-life autoencoder learns a reconstruction baseline from development windows. Its 99th-percentile reconstruction-error threshold is used as an unlabeled anomaly signal. It does not have anomaly accuracy metrics because FD001 lacks cycle-level anomaly labels.", s["body"]),
    ])
    story.extend(bullets([
        "The replay evaluates each incoming record causally and combines RUL, anomaly persistence, data quality, and model version.",
        "Deterministic triage assigns monitoring, recommendation, planned-maintenance, high-priority, or human-review outcomes.",
        "Idempotency prevents duplicate work orders for the same engine and severity. The validation replay produced 43 deduplicated mock recommendations.",
        "An LLM is not on the safety-critical decision path. A narrative assistant can explain already generated evidence but must not override a maintenance decision.",
    ], s["bullet"]))
    story.extend([
        p("7  Conclusions and recommended next work", s["h1"]),
        p("For FD001, the selected HGB pipeline is the best current choice: it achieved the strongest validation and final official-test metrics with substantially lower training and inference cost than the neural alternatives. The compact neural and autoencoder components remain useful for edge-deployment research and anomaly triage, but they should not replace the selected RUL model on the current evidence.", s["body"]),
    ])
    story.extend(bullets([
        "Validate a deployable prefix-only change-point policy before using it in labeling or in a production decision pipeline.",
        "Run FD002 and FD004 studies with condition-aware normalization and grouped engine splits.",
        "Calibrate prediction intervals and maintenance thresholds with domain owners, cost constraints, and false-alarm tolerance.",
        "Measure TinyML model size, latency, RAM use, quantization error, and update cost on the intended hardware before presenting an on-device deployment claim.",
    ], s["bullet"]))
    story.extend([
        p("8  Evidence and reproducibility", s["h1"]),
        p("The project contains reproducible runners for the validation-only study, compact sequence study, one-time official-test evaluation, and anomaly-triage replay. The full metrics matrix, model predictions, data audit, metric audit, generated figures, and unit tests are retained in the reports and artifacts directories.", s["body"]),
        p("Validation completed with 16 passing tests and one skipped optional-dependency scenario. Historical official-test scores are retained as context only; they were produced before this clean selection protocol and were not used to choose the final model.", s["body"]),
        p("References", s["h1"]),
        p("NASA C-MAPSS data documentation; Ensarioglu, Inkaya, and Emel (2023), Applied Sciences 13(21), 11893; and the user-provided research-paper materials and implementation notes.", s["ref"]),
    ])

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
