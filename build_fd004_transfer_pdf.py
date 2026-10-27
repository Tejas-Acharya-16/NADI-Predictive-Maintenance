"""Create a presentation-ready PDF summary of the FD004 transfer experiment."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "fd004_transfer"
OUT = REPORTS / "FD004_Cross_Subset_Transfer_Report.pdf"
NAVY = colors.HexColor("#17365D"); PALE = colors.HexColor("#F3F6F9"); GRID = colors.HexColor("#D9D9D9"); GREY = colors.HexColor("#595959")


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(GRID); canvas.line(doc.leftMargin, 1.25*cm, A4[0]-doc.rightMargin, 1.25*cm)
    canvas.setFillColor(GREY); canvas.setFont("Helvetica", 8); canvas.drawString(doc.leftMargin, .82*cm, "FD004 Cross Subset Transfer Report"); canvas.drawRightString(A4[0]-doc.rightMargin, .82*cm, f"Page {doc.page}"); canvas.restoreState()


def table(headers, rows, widths):
    h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=8, leading=9.5, textColor=colors.white, alignment=1)
    t = ParagraphStyle("t", fontName="Helvetica", fontSize=8.5, leading=10.5)
    n = ParagraphStyle("n", parent=t, alignment=1)
    data = [[Paragraph(f"<b>{x}</b>", h) for x in headers]] + [[Paragraph(str(x), n if i else t) for i, x in enumerate(row)] for row in rows]
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    styles = [("BACKGROUND", (0,0), (-1,0), NAVY), ("GRID", (0,0), (-1,-1), .4, GRID), ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4)]
    for index in range(1, len(data)):
        if index % 2 == 0: styles.append(("BACKGROUND", (0,index), (-1,index), PALE))
    result.setStyle(TableStyle(styles)); return result


def main():
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=5)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=11.5, leading=15, textColor=GREY, spaceAfter=3)
    meta = ParagraphStyle("meta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.7, textColor=GREY, spaceAfter=12)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.6, leading=12.8, spaceAfter=7)
    validation = pd.read_csv(REPORTS / "fd004_validation_model_comparison.csv").sort_values(["phm08_mean", "rmse"])
    transfer = pd.read_csv(REPORTS / "fd004_to_other_subsets_transfer_metrics.csv")
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7*cm, rightMargin=1.7*cm, topMargin=1.65*cm, bottomMargin=1.7*cm, title="FD004 Cross Subset Transfer Report")
    story = [
        Paragraph("FD004 Cross Subset Transfer Study", title),
        Paragraph("Train on the six-condition, two-fault FD004 subset and evaluate locked transfer to FD001, FD002, and FD003", subtitle),
        Paragraph(f"Prepared {date.today().strftime('%d %B %Y')} | CPU-bounded local execution", meta),
        Paragraph("Study design", h1),
        Paragraph("FD004 is the source because it contains six operating conditions and two degradation modes. The model selection phase uses FD004 training engines only: three engine-grouped folds and six predeclared pseudo-censoring horizons (10, 25, 50, 75, 100, and 125 cycles). The condition transformer is fitted only on FD004 settings and sensors. Target-subset labels are not used for clustering, normalization, features, or training.", body),
        Paragraph("FD004 validation selection", h1),
        table(["Candidate", "RMSE", "MAE", "R2", "PHM08 mean"], [[r.candidate, f"{r.rmse:.3f}", f"{r.mae:.3f}", f"{r.r2:.3f}", f"{r.phm08_mean:.3f}"] for r in validation.itertuples()], [8.0*cm, 2.4*cm, 2.4*cm, 1.7*cm, 2.7*cm]),
        Spacer(1, 7),
        Paragraph("The FD004-selected source model is HGB with a 100-cycle target cap and FD004-fitted condition normalization. Compared with the raw-sensor cap-125 control, condition normalization reduces validation RMSE from 19.278 to 15.323 and PHM08 mean from 9.510 to 3.460.", body),
        Paragraph("Locked cross-subset endpoint results", h1),
        table(["Target", "Engines", "RMSE", "MAE", "R2", "PHM08 mean", "Bias"], [[r.target_subset, r.target_engines, f"{r.rmse:.3f}", f"{r.mae:.3f}", f"{r.r2:.3f}", f"{r.phm08_mean:.3f}", f"{r.bias_predicted_minus_actual:.3f}"] for r in transfer.itertuples()], [2.4*cm, 2.0*cm, 2.1*cm, 2.1*cm, 1.6*cm, 2.7*cm, 2.5*cm]),
        Spacer(1, 7),
        Paragraph("Interpretation", h1),
        Paragraph("FD001 and FD003 are single-condition target domains and transfer with moderate error. FD002 is the hardest transfer target: the source model underpredicts more often and the asymmetric PHM08 penalty is high. This is evidence of domain shift, not a failure of endpoint alignment or a reason to tune on FD002 labels.", body),
        Paragraph("Recommended use", h1),
        Paragraph("Present FD004 as the final multi-condition source-model demonstration. Present the FD001–FD003 figures as honest zero-target-training transfer tests. For a production candidate in a particular target domain, validate or fine-tune only with that domain's labelled training engines, retain an engine-grouped holdout, and calibrate RUL thresholds with asset owners.", body),
        Paragraph("Reproducibility", h1),
        Paragraph("Runner: run_fd004_transfer_study.py. Detailed source/target audit, per-target endpoint predictions, split files, selection table, and serialized source model are stored in reports/fd004_transfer and artifacts/FD004/transfer.", body),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
