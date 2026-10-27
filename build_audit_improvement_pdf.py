"""Produce a concise PDF report for the completed FD001 audit-improvement study."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "Final_FD001_Audit_Improvement_Report.pdf"
NAVY = colors.HexColor("#17365D")
PALE = colors.HexColor("#F3F6F9")
GRID = colors.HexColor("#D9D9D9")
GREY = colors.HexColor("#595959")


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(GRID)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFillColor(GREY); canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, .82 * cm, "FD001 Audit and Improvement Report")
    canvas.drawRightString(A4[0] - doc.rightMargin, .82 * cm, f"Page {doc.page}")
    canvas.restoreState()


def make_table(headers, rows, widths):
    header = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8, leading=9.5, textColor=colors.white, alignment=TA_CENTER)
    text = ParagraphStyle("td", fontName="Helvetica", fontSize=8.1, leading=10)
    number = ParagraphStyle("num", parent=text, alignment=TA_CENTER)
    data = [[Paragraph(f"<b>{value}</b>", header) for value in headers]]
    for row in rows:
        data.append([Paragraph(str(value), number if idx else text) for idx, value in enumerate(row)])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    commands = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), .4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0: commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands)); return table


def main():
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.black, spaceAfter=5)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11.5, leading=15, textColor=GREY, spaceAfter=3)
    meta = ParagraphStyle("meta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.7, textColor=GREY, spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.black, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.5, spaceAfter=6)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, firstLineIndent=-8, bulletIndent=2, fontSize=9.1, leading=11.7, spaceAfter=2.5)
    caption = ParagraphStyle("caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8, leading=10, alignment=TA_CENTER, textColor=GREY, spaceAfter=5)
    summary = pd.read_csv(REPORTS / "metrics_summary.csv").sort_values(["phm08_mean", "rmse"])
    official = json.loads((REPORTS / "grouped_audit_historical_official_test_metrics.json").read_text(encoding="utf-8"))
    top = summary.head(6)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.65*cm, rightMargin=1.65*cm, topMargin=1.55*cm, bottomMargin=1.7*cm, title="FD001 Audit and Improvement Report")
    story = [
        Paragraph("NASA C MAPSS FD001 Audit and Improvement", title),
        Paragraph("Leakage-controlled grouped validation, model comparison, and competition demo readiness", subtitle),
        Paragraph(f"Prepared {date.today().strftime('%d %B %Y')} | CPU-bounded local execution", meta),
        Paragraph("Executive conclusion", h1),
        Paragraph("A five-fold engine-grouped pseudo-censoring audit was completed without loading official test files during selection. The validation champion is HGB with a 100-cycle training cap and causal 10/30-cycle features. It achieved RMSE 7.576, MAE 5.390, R2 0.946, and PHM08 mean 0.985 across 500 endpoint cases from 100 held-out engines.", body),
        Paragraph("This result applies to the predeclared 10-to-100-cycle validation horizon. It does not establish a general production cap above 100 cycles. The cap-125 HGB remains the wider-life reference, and official-benchmark results are reported separately without feeding back into selection.", body),
        Paragraph("1  Data and protocol audit", h1),
        make_table(["Data", "Rows", "Engines", "Result"], [["FD001 train", "20,631", "100", "26-column schema and all integrity checks passed"], ["FD001 test", "13,096", "100", "Endpoint alignment verified"], ["Validation", "500 cases", "100", "5 GroupKFold folds x horizons 10/25/50/75/100"]], [3.2*cm, 2.4*cm, 2.3*cm, 9.0*cm]),
        Spacer(1, 6),
        Paragraph("Features use only current and past cycles within each engine. Engine ID, final life, hidden suffixes, and official-test residuals are never input features. Cycle and cycles_seen were tested as a controlled, available-at-inference feature family; removing both degraded the cap-125 control.", body),
        Paragraph("2  Validation comparison", h1),
        make_table(["Candidate", "RMSE", "MAE", "R2", "PHM08 mean"], [[r.candidate, f"{r.rmse:.3f}", f"{r.mae:.3f}", f"{r.r2:.3f}", f"{r.phm08_mean:.3f}"] for r in top.itertuples()], [7.2*cm, 2.5*cm, 2.4*cm, 1.8*cm, 2.8*cm]),
        Spacer(1, 6),
        Paragraph("The cap-100 model won validation. The cap-125 near-constant-sensor and initial-delta variants were close alternatives. ExtraTrees and the predeclared OOF equal blend did not surpass the validation champion. CNN-LSTM remains the strongest compact neural comparison from the earlier study, but its validation RMSE was higher than the tree models.", body),
    ]
    figure = REPORTS / "figures" / "oof_actual_vs_predicted.png"
    if figure.exists():
        story.extend([Image(str(figure), width=13.8*cm, height=10.25*cm), Paragraph("Figure 1. Out-of-fold actual versus predicted RUL for the validation-selected candidate.", caption)])
    residual = REPORTS / "figures" / "oof_residual_vs_true.png"
    if residual.exists():
        story.extend([Image(str(residual), width=13.8*cm, height=10.25*cm), Paragraph("Figure 2. Out-of-fold residuals by true RUL. Values below zero are underpredictions.", caption)])
    story.append(PageBreak())
    story.extend([
        Paragraph("3  Locked historical official benchmark comparison", h1),
        Paragraph("After validation selection was locked, the cap-100 candidate was evaluated once on the already-inspected official FD001 benchmark. It is a secondary historical comparison and did not influence the feature, target, or model choice.", body),
        make_table(["Metric", "Historical benchmark value", "Interpretation"], [["RMSE", f"{official['rmse']:.3f} cycles", "Lower is better"], ["MAE", f"{official['mae']:.3f} cycles", "Lower is better"], ["R2", f"{official['r2']:.3f}", "Higher is better"], ["PHM08 mean", f"{official['phm08_mean']:.3f}", "Lower is better"], ["Bias", f"{official['bias_predicted_minus_actual']:.3f} cycles", "Negative underpredicts on average"]], [4.2*cm, 5.0*cm, 6.5*cm]),
        Spacer(1, 6),
        Paragraph("The negative bias is consistent with cap-100 targets when some official endpoints have RUL above 100. The comparison confirms the need for an expanded validation horizon schedule before choosing a production target cap. It must not be used to retroactively select cap-125 or another candidate.", body),
        Paragraph("4  Competition-ready system components", h1),
    ])
    for item in [
        "Sensor ingestion, integrity checks, causal rolling features, labelled RUL training, and saved reproducible artifacts.",
        "Compact CNN, LSTM, CNN-LSTM, TinyML-oriented, autoencoder, hybrid, and tabular model comparisons.",
        "Early-life autoencoder anomaly signal with a development-derived threshold. FD001 has no cycle-level anomaly labels, so no fabricated classification accuracy is claimed.",
        "Plain deterministic triage using RUL, anomaly persistence, data quality, and model version; it supports human review and prevents duplicate mock work orders.",
        "Streamlit dashboard driven by saved results, plus training-only data validation, descriptive drift indicators, and a two-version retraining simulation.",
    ]:
        story.append(Paragraph(item, bullet, bulletText="•"))
    story.extend([
        Paragraph("5  Recommendations", h1),
        Paragraph("Use the cap-100 result as a controlled validation finding, not a final deployment claim. Add predeclared horizons above 100 cycles and repeat grouped validation before setting a production target cap. Then extend to FD002 and FD004 with operating-condition-aware preprocessing, calibrate intervals and triage limits with domain owners, and measure TinyML latency, RAM, quantization error, and update cost on the actual edge hardware.", body),
        Paragraph("Validation status", h1),
        Paragraph("The audit generated data/pipeline/metric/protocol/baseline/ablation reports, split and endpoint files, per-candidate predictions, OOF error figures, a serialized validation-selected model, and MLOps artifacts. The test suite completed with 17 passing tests and one expected optional-dependency skip.", body),
    ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
