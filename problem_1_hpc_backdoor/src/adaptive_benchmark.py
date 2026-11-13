"""Evaluate HPC detection under hardware drift and adaptive evasion attempts.

This is intentionally a stress benchmark, not a claim that synthetic traces
are equivalent to real trojan traces. It measures whether the detector keeps
its operating point when counters are rescaled, noisy, clipped, or selectively
perturbed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import HPCBackdoorDetectorChampionship
from src.train import generate_synthetic_backdoor_traces


COUNTERS = ["cache-references", "cycles", "LLC-loads"]


def wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def transform(df, mode, seed=42):
    rng = np.random.RandomState(seed)
    result = df.copy()
    values = result[COUNTERS].to_numpy(dtype=float, copy=True)

    if mode == "global_scale":
        # Models that rely only on raw counts should be sensitive here; ratios
        # should remain stable.
        values *= rng.lognormal(mean=0.0, sigma=0.8, size=(len(values), 1))
    elif mode == "counter_noise":
        values *= rng.lognormal(mean=0.0, sigma=0.08, size=values.shape)
    elif mode == "clock_drift":
        values[:, 1] *= rng.uniform(0.65, 1.45, size=len(values))
    elif mode == "llc_evasion":
        values[:, 2] *= rng.uniform(0.90, 1.08, size=len(values))
        values[:, 1] *= rng.uniform(1.08, 1.22, size=len(values))
    elif mode == "counter_clipping":
        for column in range(values.shape[1]):
            low, high = np.percentile(values[:, column], [2, 98])
            values[:, column] = np.clip(values[:, column], low, high)
    else:
        raise ValueError(f"Unknown transform: {mode}")

    result.loc[:, COUNTERS] = np.maximum(values, 0.0)
    return result


def evaluate(detector, clean, anomalies, scenario):
    clean_pred = detector.predict(clean)
    anomaly_pred = detector.predict(anomalies)
    clean_fp = int(np.sum(clean_pred == 1))
    anomaly_tp = int(np.sum(anomaly_pred == 1))
    clean_rate = clean_fp / len(clean_pred)
    anomaly_rate = anomaly_tp / len(anomaly_pred)
    return {
        "scenario": scenario,
        "clean_fpr": clean_rate,
        "clean_fpr_ci_low": wilson_interval(clean_fp, len(clean_pred))[0],
        "clean_fpr_ci_high": wilson_interval(clean_fp, len(clean_pred))[1],
        "backdoor_tpr": anomaly_rate,
        "backdoor_tpr_ci_low": wilson_interval(anomaly_tp, len(anomaly_pred))[0],
        "backdoor_tpr_ci_high": wilson_interval(anomaly_tp, len(anomaly_pred))[1],
        "clean_samples": len(clean_pred),
        "backdoor_samples": len(anomaly_pred),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "trace.csv")
    parser.add_argument("--output", type=Path, default=root / "adaptive_results.csv")
    args = parser.parse_args()

    clean = pd.read_csv(args.input)
    fit, holdout = train_test_split(clean, test_size=0.25, random_state=123, shuffle=True)
    # Fit only on the clean training split so the adaptive benchmark remains
    # a valid held-out experiment rather than a re-test of a leaked artifact.
    detector = HPCBackdoorDetectorChampionship(target_fpr=0.01, random_state=42).fit(fit)
    anomalies = generate_synthetic_backdoor_traces(fit, n_samples=400, random_state=42)

    rows = [evaluate(detector, holdout, anomalies, "unmodified")]
    for index, mode in enumerate(
        ("global_scale", "counter_noise", "clock_drift", "llc_evasion", "counter_clipping"),
        start=1,
    ):
        rows.append(
            evaluate(
                detector,
                transform(holdout, mode, seed=100 + index),
                transform(anomalies, mode, seed=200 + index),
                mode,
            )
        )

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved adaptive benchmark: {args.output}")


if __name__ == "__main__":
    main()
