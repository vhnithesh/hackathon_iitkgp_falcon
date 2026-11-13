"""Compare raw-counter, engineered-feature, and ensemble detectors."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import HPCFeatureExtractor
from src.models import HPCBackdoorDetectorChampionship
from src.train import generate_synthetic_backdoor_traces


def evaluate(name, detector, X, y, score=None):
    pred = detector.predict(X)
    if score is None:
        score = -detector.decision_function(X)
    return {
        "detector": name,
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "auroc": roc_auc_score(y, score),
        "fpr": np.mean(pred[: np.sum(y == 0)] == 1),
        "tpr": np.mean(pred[np.sum(y == 0):] == 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "trace.csv")
    parser.add_argument("--output", type=Path, default=root / "ablation_results.csv")
    args = parser.parse_args()

    clean = pd.read_csv(args.input)
    fit, holdout = train_test_split(clean, test_size=0.25, random_state=123, shuffle=True)
    anomalies = generate_synthetic_backdoor_traces(fit, n_samples=400, random_state=42)
    X = pd.concat([holdout, anomalies], ignore_index=True)
    y = np.array([0] * len(holdout) + [1] * len(anomalies))

    rows = []
    raw_scaler = StandardScaler().fit(fit)
    raw_detector = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    raw_detector.fit(raw_scaler.transform(fit))
    rows.append(
        evaluate(
            "raw_counters_iforest",
            _WrappedDetector(raw_detector, raw_scaler),
            X,
            y,
            score=-raw_detector.decision_function(raw_scaler.transform(X)),
        )
    )

    feature_extractor = HPCFeatureExtractor().fit(fit)
    feature_scaler = StandardScaler().fit(feature_extractor.transform(fit))
    engineered_detector = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    engineered_detector.fit(feature_scaler.transform(feature_extractor.transform(fit)))
    rows.append(
        evaluate(
            "engineered_iforest",
            _WrappedDetector(engineered_detector, feature_scaler, feature_extractor),
            X,
            y,
            score=-engineered_detector.decision_function(
                feature_scaler.transform(feature_extractor.transform(X))
            ),
        )
    )

    ensemble = HPCBackdoorDetectorChampionship(target_fpr=0.01, random_state=42).fit(fit)
    rows.append(evaluate("full_ensemble", ensemble, X, y, score=ensemble.decision_function(X)))

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved ablation study: {args.output}")


class _WrappedDetector:
    def __init__(self, detector, scaler, extractor=None):
        self.detector = detector
        self.scaler = scaler
        self.extractor = extractor

    def _transform(self, X):
        if self.extractor is not None:
            X = self.extractor.transform(X)
        return self.scaler.transform(X)

    def predict(self, X):
        # IsolationForest returns +1 (inlier) / -1 (outlier); map to 0/1 labels.
        return (self.detector.predict(self._transform(X)) == -1).astype(int)

    def decision_function(self, X):
        return self.detector.decision_function(self._transform(X))


if __name__ == "__main__":
    main()
