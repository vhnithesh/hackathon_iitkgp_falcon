import sys
import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import HPCBackdoorDetectorChampionship

def generate_synthetic_backdoor_traces(clean_df, n_samples=400, random_state=42):
    rng = np.random.RandomState(random_state)
    anomalies = []
    
    stds = clean_df.std().values

    # Profile 1: Elevated LLC loads and cache references (Memory & activation expansion)
    n1 = n_samples // 4
    p1 = clean_df.sample(n1, replace=True, random_state=rng.randint(1000)).to_numpy(copy=True)
    p1[:, 0] *= rng.uniform(1.08, 1.25, size=n1)
    p1[:, 1] *= rng.uniform(1.05, 1.20, size=n1)
    p1[:, 2] *= rng.uniform(1.15, 1.40, size=n1)
    anomalies.append(p1)

    # Profile 2: Execution stalls (cycle explosion with reduced/flat cache events)
    n2 = n_samples // 4
    p2 = clean_df.sample(n2, replace=True, random_state=rng.randint(1000)).to_numpy(copy=True)
    p2[:, 1] *= rng.uniform(1.15, 1.35, size=n2)
    p2[:, 2] *= rng.uniform(0.85, 1.10, size=n2)
    anomalies.append(p2)

    # Profile 3: Cache thrashing / LLC load surge
    n3 = n_samples // 4
    p3 = clean_df.sample(n3, replace=True, random_state=rng.randint(1000)).to_numpy(copy=True)
    p3[:, 0] *= rng.uniform(0.95, 1.05, size=n3)
    p3[:, 2] *= rng.uniform(1.20, 1.50, size=n3)
    anomalies.append(p3)

    # Profile 4: Low-overhead stealthy backdoor (Subtle covariance break)
    n4 = n_samples - (n1 + n2 + n3)
    p4 = clean_df.sample(n4, replace=True, random_state=rng.randint(1000)).to_numpy(copy=True)
    p4[:, 0] += rng.normal(3.0 * stds[0], stds[0] * 0.5, size=n4)
    p4[:, 1] += rng.normal(3.0 * stds[1], stds[1] * 0.5, size=n4)
    p4[:, 2] += rng.normal(3.5 * stds[2], stds[2] * 0.5, size=n4)
    anomalies.append(p4)

    return pd.DataFrame(np.vstack(anomalies), columns=clean_df.columns)

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "trace.csv")
    print(f"Loading training traces from: {os.path.abspath(data_path)}")
    df_clean = pd.read_csv(data_path)
    print(f"Clean traces loaded: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")

    print("\nTraining ensemble HPC backdoor detector...")
    print("- Deep Autoencoder (PyTorch, 50 epochs)")
    print("- Isolation Forest (200 trees)")
    print("- One-Class SVM (RBF kernel)")
    print("- ECOD (PyOD Empirical Cumulative Distribution)")
    print("- Gaussian Mixture Model (full covariance)")
    print("- FastMCD Elliptic Envelope (0.92 covariance)")

    # Keep a clean holdout completely separate from fitting and threshold
    # calibration. Evaluating rows used during fitting would underestimate FPR.
    df_fit, df_holdout = train_test_split(
        df_clean, test_size=0.25, random_state=123, shuffle=True
    )

    detector = HPCBackdoorDetectorChampionship(target_fpr=0.01, random_state=42)
    detector.fit(df_fit)
    print(f"Model trained! Calibrated Decision Threshold: {detector.threshold:.5f}")

    clean_preds = detector.predict(df_holdout)
    clean_fpr = np.mean(clean_preds == 1)
    print(f"Clean Holdout FPR (False Alarm Rate): {clean_fpr * 100:.2f}% (Target <= 1.0%)")

    print("\nEvaluating on Synthetic Backdoor Benchmark (200 clean holdout + 400 backdoor traces)...")
    df_synthetic_backdoor = generate_synthetic_backdoor_traces(df_fit, n_samples=400, random_state=42)

    X_val = pd.concat([df_holdout, df_synthetic_backdoor], ignore_index=True)
    y_val = np.array([0] * len(df_holdout) + [1] * len(df_synthetic_backdoor))

    val_preds = detector.predict(X_val)
    val_scores = detector.decision_function(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0, 1]).ravel()
    acc = accuracy_score(y_val, val_preds)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    f1 = f1_score(y_val, val_preds, zero_division=0)
    auroc = roc_auc_score(y_val, val_scores)

    print("=" * 60)
    print("SYNTHETIC HOLDOUT BENCHMARK RESULTS:")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"TPR      : {tpr * 100:.2f}% (Target > 90%)")
    print(f"FPR      : {fpr * 100:.2f}% (Target < 10%)")
    print(f"F1 Score : {f1:.4f} (Target > 0.90)")
    print(f"AUROC    : {auroc:.4f} (Target > 0.95)")
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 60)

    save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "detector_model.pkl")
    joblib_path = os.path.join(save_dir, "hpc_backdoor_detector.joblib")

    # Keep the holdout benchmark honest, then use every supplied benign trace
    # for the deployment artifact. The threshold is the 99th percentile of the
    # ensemble score on these same training rows (see models.py), so the clean
    # FPR on unseen traces is higher than 1%: about 4% on the holdout above.
    print("\nRefitting final submission model on all supplied benign traces...")
    final_detector = HPCBackdoorDetectorChampionship(
        target_fpr=0.01, random_state=42
    )
    final_detector.fit(df_clean)

    with open(pkl_path, "wb") as f:
        pickle.dump(final_detector, f)
    joblib.dump(final_detector, joblib_path)

    print(f"\nModel saved successfully:")
    print(f"1. {os.path.abspath(pkl_path)}")
    print(f"2. {os.path.abspath(joblib_path)}")

if __name__ == "__main__":
    main()
