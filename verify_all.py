"""Run the reproducibility audit for both hackathon problems.

The Problem 1 benchmark is deliberately fit on a clean training split and
measured on a disjoint clean holdout. Problem 2 is audited from PNG files
reloaded from disk, so quantization is included in the measurement.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
P1_ROOT = ROOT / "problem_1_hpc_backdoor"
P2_ROOT = ROOT / "problem_2_cnn_adversarial"
sys.path.insert(0, str(P1_ROOT))
sys.path.insert(0, str(P2_ROOT))

from src.models import HPCBackdoorDetectorChampionship  # noqa: E402
from src.train import generate_synthetic_backdoor_traces  # noqa: E402
from audit_adversarial import audit as audit_problem_2  # noqa: E402


P1_TARGETS = {"acc": 0.90, "tpr": 0.90, "fpr": 0.10, "f1": 0.90, "auroc": 0.95}


def verify_problem_1():
    print("\n" + "=" * 65)
    print("AUDITING PROBLEM 1: HPC BACKDOOR ANOMALY DETECTION")
    print("=" * 65)

    model_path = P1_ROOT / "saved_models" / "hpc_backdoor_detector.joblib"
    csv_path = P1_ROOT / "trace.csv"
    if not model_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing Problem 1 artifact or input: {model_path}, {csv_path}")

    # Smoke-test the submitted artifact, then independently validate the
    # training procedure on a disjoint split. This avoids reporting a training
    # set score as if it were an unseen-data result.
    submitted_detector = joblib.load(model_path)
    df_clean = pd.read_csv(csv_path)
    submitted_detector.predict(df_clean.iloc[:1])

    df_fit, df_holdout = train_test_split(
        df_clean, test_size=0.25, random_state=123, shuffle=True
    )
    detector = HPCBackdoorDetectorChampionship(target_fpr=0.01, random_state=42)
    detector.fit(df_fit)

    df_backdoor = generate_synthetic_backdoor_traces(
        df_fit, n_samples=400, random_state=42
    )
    X_val = pd.concat([df_holdout, df_backdoor], ignore_index=True)
    y_val = np.array([0] * len(df_holdout) + [1] * len(df_backdoor))
    val_preds = detector.predict(X_val)
    val_scores = detector.decision_function(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
    metrics = {
        "acc": accuracy_score(y_val, val_preds),
        "tpr": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "f1": f1_score(y_val, val_preds, zero_division=0),
        "auroc": roc_auc_score(y_val, val_scores),
    }
    passed = (
        metrics["acc"] > P1_TARGETS["acc"]
        and metrics["tpr"] > P1_TARGETS["tpr"]
        and metrics["fpr"] < P1_TARGETS["fpr"]
        and metrics["f1"] > P1_TARGETS["f1"]
        and metrics["auroc"] > P1_TARGETS["auroc"]
    )

    print(f"Accuracy : {metrics['acc'] * 100:.2f}%  (Target: > 90%)")
    print(f"TPR      : {metrics['tpr'] * 100:.2f}%  (Target: > 90%)")
    print(f"FPR      : {metrics['fpr'] * 100:.2f}%  (Target: < 10%)")
    print(f"F1-Score : {metrics['f1']:.4f}  (Target: > 0.90)")
    print(f"AUROC    : {metrics['auroc']:.4f}  (Target: > 0.95)")
    print(f"Benchmark Status: {'PASSED' if passed else 'FAILED'}")
    metrics["passed"] = passed
    return metrics


def verify_problem_2():
    return audit_problem_2(
        P2_ROOT / "adv_images",
        P2_ROOT / "p2_data" / "clean_images",
        P2_ROOT / "p2_data" / "cnn_model.pt",
    )


def main():
    t0 = time.time()
    print("=" * 65)
    print("IIT KHARAGPUR ML SECURITY HACKATHON — MASTER SUBMISSION AUDIT")
    print("=" * 65)
    p1 = verify_problem_1()
    p2 = verify_problem_2()
    passed = bool(p1["passed"] and p2["passed"])

    print("\n" + "=" * 65)
    print(f"MASTER AUDIT COMPLETE IN {time.time() - t0:.2f} SECONDS")
    print("=" * 65)
    print(
        f"Problem 1 — AUROC: {p1['auroc']:.4f} | F1: {p1['f1']:.4f} | "
        f"Accuracy: {p1['acc'] * 100:.2f}% | FPR: {p1['fpr'] * 100:.2f}%"
    )
    print(
        f"Problem 2 — ASR: {p2['asr'] * 100:.2f}% | SSIM: {p2['ssim']:.4f} | "
        f"PSNR: {p2['psnr']:.2f} dB | L2: {p2['l2']:.4f}"
    )
    print(f"STATUS    : {'ALL HARD METRICS PASSED' if passed else 'ONE OR MORE HARD METRICS FAILED'}")
    print("=" * 65)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
