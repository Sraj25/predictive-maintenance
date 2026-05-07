"""
================================================
 Predictive Maintenance - EDA & Preprocessing
================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path="data/sensor_data.csv"):
    df = pd.read_csv(path)
    print("=" * 50)
    print(" DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape       : {df.shape}")
    print(f"Missing vals:\n{df.isnull().sum()}\n")
    print(df.describe().round(2))
    return df


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    """Fill missing values with column medians (robust to outliers)."""
    features = ["temperature", "vibration", "pressure", "runtime_hours"]
    for col in features:
        median_val = df[col].median()
        n_missing  = df[col].isnull().sum()
        if n_missing:
            df[col] = df[col].fillna(median_val)
            print(f"[✓] Filled {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # Clip extreme outliers beyond 3 standard deviations
    for col in features:
        mean, std = df[col].mean(), df[col].std()
        df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)

    print(f"\n[✓] Cleaned dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────
def plot_distributions(df):
    features = ["temperature", "vibration", "pressure", "runtime_hours"]
    colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Feature Distributions by Failure Status", fontsize=16, fontweight="bold", y=1.01)

    for ax, feat, color in zip(axes.flat, features, colors):
        for status, label, ls in [(0, "No Failure", "-"), (1, "Failure", "--")]:
            subset = df[df["failure"] == status][feat]
            ax.hist(subset, bins=40, alpha=0.55, label=label,
                    color=color if status == 0 else "crimson", edgecolor="white")
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/feature_distributions.png")


# ─────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────
def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 11})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/correlation_heatmap.png")


# ─────────────────────────────────────────────
# 5. BOXPLOT — OUTLIER DETECTION
# ─────────────────────────────────────────────
def plot_boxplots(df):
    features = ["temperature", "vibration", "pressure", "runtime_hours"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Boxplots — Outlier Detection", fontsize=14, fontweight="bold")

    for ax, feat in zip(axes, features):
        df.boxplot(column=feat, by="failure", ax=ax,
                   boxprops=dict(color="#4C72B0"),
                   medianprops=dict(color="crimson", linewidth=2))
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("Failure (0=No, 1=Yes)")

    plt.suptitle("")          # remove auto title from boxplot
    fig.suptitle("Boxplots — Outlier Detection per Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/boxplots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/boxplots.png")


# ─────────────────────────────────────────────
# 6. PAIRPLOT
# ─────────────────────────────────────────────
def plot_pairplot(df):
    sample = df.sample(min(1000, len(df)), random_state=42)
    sample["Failure"] = sample["failure"].map({0: "No Failure", 1: "Failure"})
    pp = sns.pairplot(sample, hue="Failure", vars=["temperature","vibration","pressure","runtime_hours"],
                      plot_kws={"alpha": 0.4}, diag_kind="kde",
                      palette={"No Failure": "#4C72B0", "Failure": "crimson"})
    pp.fig.suptitle("Pairplot of Sensor Features", y=1.02, fontsize=14, fontweight="bold")
    plt.savefig(f"{PLOTS_DIR}/pairplot.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/pairplot.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    plot_distributions(df)
    plot_correlation(df)
    plot_boxplots(df)
    plot_pairplot(df)
    # Save cleaned data for model training
    df.to_csv("data/sensor_data_cleaned.csv", index=False)
    print("\n[✓] Cleaned data saved → data/sensor_data_cleaned.csv")