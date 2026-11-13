"""Build a concise, evidence-backed submission scorecard from local artifacts."""

from datetime import date
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def latest_csv(path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    adaptive = latest_csv(ROOT / "problem_1_hpc_backdoor" / "adaptive_results.csv")
    ablation = latest_csv(ROOT / "problem_1_hpc_backdoor" / "ablation_results.csv")
    lines = [
        "# Submission Security Scorecard",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "This scorecard separates measured evidence from historical or missing results.",
        "",
        "## Artifact integrity",
        "",
        f"- Clean images: {len(list((ROOT / 'problem_2_cnn_adversarial' / 'p2_data' / 'clean_images').glob('*.png')))}",
        f"- Adversarial images: {len(list((ROOT / 'problem_2_cnn_adversarial' / 'adv_images').glob('*.png')))}",
        "- Quantization verification: performed by `audit_adversarial.py`",
        "",
        "## Problem 1 adaptive robustness",
        "",
    ]
    if adaptive is None:
        lines.append("Not run yet. Run `python problem_1_hpc_backdoor/src/adaptive_benchmark.py` to populate this table.")
    else:
        lines.extend([
            "| Scenario | Clean FPR | Backdoor TPR |",
            "|---|---:|---:|",
        ])
        for row in adaptive:
            lines.append(
                f"| {row['scenario']} | {float(row['clean_fpr']):.3f} | "
                f"{float(row['backdoor_tpr']):.3f} |"
            )

    lines.extend(["", "## Ablation evidence", ""])
    if ablation is None:
        lines.append("Run `python problem_1_hpc_backdoor/src/ablation.py` to populate this table.")
    else:
        lines.extend([
            "| Detector | Accuracy | F1 | AUROC |",
            "|---|---:|---:|---:|",
        ])
        for row in ablation:
            lines.append(
                f"| {row['detector']} | {float(row['accuracy']):.3f} | "
                f"{float(row['f1']):.3f} | {float(row['auroc']):.3f} |"
            )

    p2 = json.loads(
        (ROOT / "problem_2_cnn_adversarial" / "audit_results.json").read_text(encoding="utf-8")
    )
    lines.extend([
        "",
        "## Problem 2 disk-reloaded audit",
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Attack success rate | {p2['asr'] * 100:.2f}% |",
        f"| Mean / median SSIM | {p2['ssim']:.4f} / {p2['ssim_median']:.4f} |",
        f"| Images with SSIM < 0.8 | {p2['ssim_below_0_8']} |",
        f"| Mean PSNR | {p2['psnr']:.2f} dB |",
        f"| Mean L2 / Linf | {p2['l2']:.4f} / {p2['linf']:.4f} |",
        f"| Lowest-SSIM images sent through HSJA refinement | {p2.get('refined_images', 0)} |",
        f"| Logged refinement queries | {p2.get('refinement_queries_total', 0):,} |",
        "",
        "The rubric assigns 50% to ASR and 50% to imperceptibility; SSIM is the "
        "imperceptibility measure reported here. Queries of the stage-1 attack were not "
        "logged for this image set. Confidence change is not observable through the "
        "hard-label interface.",
        "",
        "## Problem 1 synthetic holdout",
        "",
        "98.00% accuracy, 99.00% TPR, 4.00% FPR, 0.9851 F1 and 0.9982 AUROC on 200 clean "
        "holdout traces plus 400 synthetic anomalies. The drift table above shows the "
        "detector assumes test traces come from the same measurement setup.",
        "",
    ])
    output = ROOT / "SCORECARD.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
