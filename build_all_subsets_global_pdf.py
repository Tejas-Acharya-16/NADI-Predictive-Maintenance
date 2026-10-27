"""Build a concise PDF from the completed global C-MAPSS study outputs."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "all_subsets_global"
OUT = REPORTS / "Global_Train_All_Test_All_Report.pdf"
NAVY = colors.HexColor("#17365D")
PALE = colors.HexColor("#F3F6F9")
GRID = colors.HexColor("#D9D9D9")
GREY = colors.HexColor("#595959")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(doc.leftMargin, 1.25 * cm, A4[0] - doc.rightMargin, 1.25 * cm)
    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(doc.leftMargin, 0.82 * cm, "Global C-MAPSS RUL Study")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.82 * cm, f"Page {doc.page}")
    canvas.restoreState()


def make_table(headers, rows, widths):
    header = ParagraphStyle("header", fontName="Helvetica-Bold", fontSize=7.8, leading=9, textColor=colors.white, alignment=1)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=8.2, leading=10)
    centre = ParagraphStyle("centre", parent=body, alignment=1)
    data = [[Paragraph(f"<b>{column}</b>", header) for column in headers]]
    data += [[Paragraph(str(value), body if index == 0 else centre) for index, value in enumerate(row)] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), PALE))
    table.setStyle(TableStyle(style))
    return table


def main() -> None:
    comparison = pd.read_csv(REPORTS / "global_validation_model_comparison.csv").sort_values(["phm08_mean", "rmse"])
    test = pd.read_csv(REPORTS / "official_test_metrics_by_subset.csv").sort_values("subset")
    pooled = pd.read_csv(REPORTS / "official_test_metrics_pooled.csv").iloc[0]
    selected = comparison.loc[comparison["selected_by_global_validation"]].iloc[0]
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, spaceAfter=5)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11.3, leading=15, textColor=GREY, spaceAfter=3)
    meta = ParagraphStyle("meta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.7, textColor=GREY, spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=12.7, spaceAfter=7)
    document = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.65 * cm, rightMargin=1.65 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm, title="Global Train All Test All C-MAPSS RUL Report")
    story = [
        Paragraph("Global C-MAPSS RUL Study", title),
        Paragraph("Train on every FD001–FD004 training trajectory; evaluate the locked model on every official FD001–FD004 test set", subtitle),
        Paragraph(f"Prepared {date.today().strftime('%d %B %Y')} | CPU-bounded local execution | 709 training engines | 707 official test engines", meta),
        Paragraph("Protocol", h1),
        Paragraph("The four training files were concatenated after remapping engine identifiers. Three HistGradientBoosting candidates were selected using only combined-training data: three subset-stratified engine-grouped folds and fixed 10/25/50/75/100/125-cycle pseudo-censored endpoints. The condition-normalized candidates fit six operating-condition clusters and sensor moments on development rows in each fold. Official test labels were read only after the validation choice was fixed.", body),
        Paragraph("Global validation selection", h1),
        make_table(
            ["Candidate", "Cap", "Condition norm.", "RMSE", "MAE", "R2", "PHM08 mean"],
            [[row.candidate, row.rul_cap, "yes" if row.condition_normalized else "no", f"{row.rmse:.3f}", f"{row.mae:.3f}", f"{row.r2:.3f}", f"{row.phm08_mean:.3f}"] for row in comparison.itertuples()],
            [7.25 * cm, 1.2 * cm, 2.2 * cm, 1.75 * cm, 1.75 * cm, 1.35 * cm, 2.3 * cm],
        ),
        Spacer(1, 7),
        Paragraph(f"Selected before official testing: <b>{selected.candidate}</b>. It is ranked by mean PHM08, followed by RMSE. The cap and normalizer are then refit on all 709 training engines before endpoint prediction.", body),
        Paragraph("Locked official-test results", h1),
        make_table(
            ["Test subset", "Engines", "RMSE", "MAE", "R2", "PHM08 mean", "Bias"],
            [[row.subset, row.engines, f"{row.rmse:.3f}", f"{row.mae:.3f}", f"{row.r2:.3f}", f"{row.phm08_mean:.3f}", f"{row.bias_predicted_minus_actual:.3f}"] for row in test.itertuples()],
            [2.1 * cm, 1.65 * cm, 1.75 * cm, 1.75 * cm, 1.25 * cm, 2.3 * cm, 1.7 * cm],
        ),
        Spacer(1, 7),
        Paragraph("Pooled result", h1),
        Paragraph(f"Across all {int(pooled.engines)} official test engines: RMSE <b>{pooled.rmse:.3f}</b>, MAE <b>{pooled.mae:.3f}</b>, R2 <b>{pooled.r2:.3f}</b>, and PHM08 mean <b>{pooled.phm08_mean:.3f}</b>. This aggregate is descriptive; the per-subset rows remain the primary result because fault-mode and operating-condition coverage differs across FD001–FD004.", body),
        Paragraph("Interpretation and next step", h1),
        Paragraph("This is the requested all-training-data model, evaluated consistently on every official test file. It measures performance after exposure to both single- and multi-condition data and both one- and two-fault training regimes. It should be compared with the earlier FD004-only transfer study as a separate experiment, because that study intentionally withheld FD001–FD003 training data. The saved endpoint predictions support further error analysis, calibration review, or a lightweight edge-model distillation experiment.", body),
        Paragraph("Reproducibility", h1),
        Paragraph("Runner: run_all_subsets_global_study.py. The complete validation comparison, per-subset endpoint predictions, global model artifact, split inventories, and machine-readable metrics are stored in reports/all_subsets_global and artifacts/all_subsets_global.", body),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    arguments = parser.parse_args()
    REPORTS = arguments.report_dir
    OUT = REPORTS / "Global_Train_All_Test_All_Report.pdf"
    main()
