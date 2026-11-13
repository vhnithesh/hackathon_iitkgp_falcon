import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(description="Evaluate the HPC backdoor detector on a test CSV.")
    parser.add_argument("--input", "-i", type=str, default="trace.csv", help="Path to input trace CSV file")
    parser.add_argument("--model", "-m", type=str, default="saved_models/hpc_backdoor_detector.joblib", help="Path to saved model")
    parser.add_argument("--output", "-o", type=str, default="predictions.csv", help="Path to output predictions CSV")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, args.model) if not os.path.isabs(args.model) else args.model
    # Relative input paths are tried from the current directory first, then
    # from problem_1_hpc_backdoor/, so the script works from any directory.
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        input_path = os.path.join(base_dir, args.input)
    output_path = os.path.abspath(args.output)

    print(f"Loading detector from: {model_path}")
    detector = joblib.load(model_path)

    print(f"Loading input traces from: {input_path}")
    df = pd.read_csv(input_path)

    preds = detector.predict(df)
    scores = detector.decision_function(df)
    probs = detector.predict_proba(df)

    results_df = pd.DataFrame({
        "sample_id": np.arange(len(preds)),
        "predicted_label": preds,
        "label_name": ["Clean" if p == 0 else "Backdoor" for p in preds],
        "anomaly_score": scores,
        "prob_backdoor": probs[:, 1]
    })

    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Total samples: {len(preds)}")
    print(f"Clean detected: {np.sum(preds == 0)} ({np.mean(preds == 0) * 100:.2f}%)")
    print(f"Backdoor detected: {np.sum(preds == 1)} ({np.mean(preds == 1) * 100:.2f}%)")

if __name__ == "__main__":
    main()
