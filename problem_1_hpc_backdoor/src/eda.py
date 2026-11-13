import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_eda(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Dataset Shape: {df.shape[0]} rows (clean inferences) x {df.shape[1]} columns")
    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
    print("\nSkewness:")
    print(df.skew())

    # 1. Correlation Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".3f", cbar_kws={'label': 'Pearson Correlation'})
    plt.title("HPC Feature Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_plot = os.path.join(output_dir, "hpc_correlation_heatmap.png")
    plt.savefig(corr_plot, dpi=200)
    plt.close()
    print(f"Saved correlation heatmap to: {corr_plot}")

    # 2. PCA 2D Scatter to visualize cluster tightness
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c="#1A237E", alpha=0.6, edgecolors="none", s=25)
    plt.title(f"PCA 2D Projection of Clean HPC Traces\n(Explained Variance: {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pca_plot = os.path.join(output_dir, "hpc_pca_scatter.png")
    plt.savefig(pca_plot, dpi=200)
    plt.close()
    print(f"Saved PCA scatter plot to: {pca_plot}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "trace.csv")
    out_dir = base_dir
    run_eda(csv_path, out_dir)
