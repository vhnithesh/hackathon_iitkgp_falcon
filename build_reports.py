"""Build concise, reproducible PDF reports for both competition tracks."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pdf"
NAVY = colors.HexColor("#15345B")
BLUE = colors.HexColor("#2367A8")
PALE = colors.HexColor("#EAF1F8")
GRAY = colors.HexColor("#536171")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=22, leading=27, textColor=NAVY, alignment=TA_CENTER,
    spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="Deck", parent=styles["Normal"], fontSize=10, leading=15,
    textColor=GRAY, alignment=TA_CENTER, spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=13, leading=17, textColor=BLUE, spaceBefore=4 * mm,
    spaceAfter=2 * mm, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BodyCompact", parent=styles["BodyText"], fontSize=9.4,
    leading=14, spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    name="Cell", parent=styles["BodyText"], fontSize=8.7, leading=12,
))
styles.add(ParagraphStyle(
    name="SmallNote", parent=styles["BodyText"], fontSize=8,
    leading=11, textColor=GRAY, spaceBefore=2 * mm,
))


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#D8E0E8"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(18 * mm, 10 * mm, "IIT Kharagpur ML Security Hackathon 2026")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def make_doc(filename, title, deck, sections):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=22 * mm,
        title=title, author="Hackathon Submission Team",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])
    story = [Paragraph(title, styles["ReportTitle"]), Paragraph(deck, styles["Deck"])]
    for kind, content in sections:
        if kind == "heading":
            story.append(Paragraph(content, styles["Section"]))
        elif kind == "body":
            story.append(Paragraph(content, styles["BodyCompact"]))
        elif kind == "table":
            rows = [[Paragraph(str(cell), styles["Cell"]) for cell in row] for row in content]
            col_width = doc.width / len(content[0])
            table = Table(rows, colWidths=[col_width] * len(content[0]), repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), PALE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CAD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 3 * mm)])
        elif kind == "note":
            story.append(Paragraph(content, styles["SmallNote"]))
    doc.build(story)
    print(path)


make_doc(
    "Problem_1_HPC_Backdoor_Detection.pdf",
    "Problem 1 | HPC Backdoor Detection",
    "Unsupervised anomaly detection from benign hardware performance counter traces",
    [
        ("heading", "Objective"),
        ("body", "Detect inference traces that differ from benign behavior using only the provided clean hardware performance counter (HPC) traces for training. No backdoor traces are used for fitting."),
        ("heading", "Method"),
        ("body", "The pipeline validates the required cache-references, cycles, and LLC-loads counters, then derives normalized ratios and logarithmic counter features. A six-component anomaly ensemble combines an autoencoder, Isolation Forest, One-Class SVM, ECOD, Gaussian mixture model, and robust covariance detector. The ensemble threshold is calibrated from benign training scores."),
        ("body", "For an honest internal benchmark, 600 benign rows are used to fit the evaluation detector and 200 disjoint benign rows are held out. Four hundred synthetic anomalies are generated from separate perturbation profiles representing elevated cache activity, execution stalls, cache-load surges, and subtle counter shifts. After the benchmark, the final submission pipeline refits on all 800 benign rows."),
        ("heading", "Measured synthetic holdout results"),
        ("table", [["Metric", "Target", "Result"], ["AUROC", "&gt; 0.95", "0.9982"], ["F1 score", "&gt; 0.90", "0.9851"], ["True positive rate", "&gt; 90%", "99.00%"], ["False positive rate", "&lt; 10%", "4.00%"], ["Accuracy", "-", "98.00%"]]),
        ("body", "Ablation on the same split: Isolation Forest on raw counters reaches AUROC 0.991 with 7.5% FPR, on engineered features AUROC 0.991 with 5.0% FPR, and the full ensemble AUROC 0.998 with 4.0% FPR. Shifts of 5% or more in any single counter are separated with AUROC of at least 0.95."),
        ("heading", "Reproducibility and limitations"),
        ("body", "Run <b>python problem_1_hpc_backdoor/src/train.py</b> to reproduce the split benchmark and write the final model files under <b>problem_1_hpc_backdoor/saved_models/</b>. Run <b>python verify_all.py</b> for the cross-track audit. The reported anomaly metrics are measured on synthetic proxies, not real backdoor traces or the private evaluation set; they should not be interpreted as guaranteed competition performance. The detector models the benign distribution of this specific measurement setup: in a drift stress test (global rescaling, counter noise, clock drift) almost all clean traces are flagged, so test traces must come from the same setup, or the detector must be refit on clean traces from the new one."),
        ("note", "Evaluation metrics are internal reproducibility results. The organizer's private clean/backdoor validation traces remain the authoritative test."),
    ],
)

P2 = json.loads((ROOT / "problem_2_cnn_adversarial" / "audit_results.json").read_text(encoding="utf-8"))

make_doc(
    "Problem_2_Black_Box_Adversarial_Attack.pdf",
    "Problem 2 | Black-Box Adversarial Attack",
    "Hard-label decision-based attack against the supplied pretrained CNN",
    [
        ("heading", "Objective and constraints"),
        ("body", "Create one visually similar adversarial PNG for each of the 1,000 supplied clean images. The attack treats the model as a hard-label oracle: it queries predicted labels only and never uses the architecture, parameters, gradients, logits, or probabilities."),
        ("heading", "Attack method"),
        ("body", "<b>Stage 1 - directional search.</b> Smooth random directions (bilinear-upsampled 8x8, 12x12 and 16x16 noise) are tried at increasing radii; for the directions that flip the label, boundary bisection finds the closest crossing. A checked cross-class anchor image is the fallback. Candidates are rounded to 8-bit and re-queried, so every image is adversarial as saved."),
        ("body", "<b>Stage 2 - bisection toward the clean image</b> in 8-bit space, keeping only label-changing points."),
        ("body", f"<b>Stage 3 - HopSkipJump-style refinement</b> of the {P2.get('refined_images', 0)} lowest-SSIM images. At the boundary point, the hard labels of random probes give a Monte-Carlo estimate of the boundary normal; the attack steps along it and bisects back toward the clean image (15 iterations, up to 60 probes each). A refined image replaces the previous one only if it is still adversarial after 8-bit rounding and has higher SSIM."),
        ("heading", "Disk-reloaded audit"),
        ("table", [
            ["Measure", "Result", "Interpretation"],
            ["Attack success rate", f"{P2['asr'] * 100:.2f}% ({round(P2['asr'] * P2['total'])}/{P2['total']})", "Saved PNGs reloaded and re-classified"],
            ["Mean SSIM", f"{P2['ssim']:.4f}", f"Median {P2['ssim_median']:.4f}; {P2['ssim_below_0_8']} images below 0.8"],
            ["Mean PSNR", f"{P2['psnr']:.2f} dB", "Higher is less visible"],
            ["Mean L2 / Linf", f"{P2['l2']:.4f} / {P2['linf']:.4f}", f"Median Linf {P2['linf_median_255']:.0f}/255"],
            ["Stage-3 oracle queries", f"{P2.get('refinement_queries_total', 0):,}", f"Mean {P2.get('refinement_queries_mean', 0):.0f} per refined image (logged)"],
        ]),
        ("heading", "Scoring alignment and caveats"),
        ("body", "The rubric weights attack success and imperceptibility equally; SSIM is reported as the imperceptibility measure. Stage-3 queries are logged per image in <b>hsja_queries.csv</b>; queries of the earlier stages were not logged for this image set. Those stage-1 images came from earlier versions of the directional search (the 25 worst were re-searched with targeted_search.py); the notebook's REGENERATE path reruns the current pipeline end to end and logs every query. Confidence change cannot be observed through a hard-label oracle."),
        ("heading", "Reproduction"),
        ("body", "<b>p2_code.ipynb</b> contains the complete attack (stages 1 and 3) and audits the saved images; set REGENERATE = True to rebuild all images from scratch. The scripts <b>attack.py</b>, <b>refine_existing.py</b>, <b>refine_worst.py</b> and <b>audit_adversarial.py</b> run each step separately."),
    ],
)
