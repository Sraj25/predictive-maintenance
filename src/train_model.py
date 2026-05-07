"""
================================================
 Predictive Maintenance - Model Training
 Trains Logistic Regression & Random Forest,
 evaluates both, and saves the best model.
================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (accuracy_score, confusion_matrix,
                                       classification_report, roc_auc_score,
                                       ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
PLOTS_DIR = "plots"
MODEL_DIR = "model"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

FEATURES = ["temperature", "vibration", "pressure", "runtime_hours"]
TARGET   = "failure"


# ─────────────────────────────────────────────
# 1. LOAD & SPLIT
# ─────────────────────────────────────────────
def load_and_split(path="data/sensor_data_cleaned.csv", test_size=0.20):
    df = pd.read_csv(path)
    X  = df[FEATURES]
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"[✓] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"    Failure rate — Train: {y_train.mean()*100:.1f}%  Test: {y_test.mean()*100:.1f}%")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. SCALE FEATURES
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("[✓] Scaler saved → model/scaler.pkl")
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


# ─────────────────────────────────────────────
# 4. EVALUATE A MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name, X_train=None, y_train=None):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc    = accuracy_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["No Failure", "Failure"])

    print("\n" + "=" * 55)
    print(f"  {model_name}")
    print("=" * 55)
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {roc:.4f}")
    print("\n  Classification Report:")
    print(report)

    # Cross-val score on training set
    if X_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=StratifiedKFold(5), scoring="roc_auc")
        print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Confusion matrix plot
    _plot_confusion_matrix(y_test, y_pred, model_name)

    return {"model": model_name, "accuracy": acc, "roc_auc": roc}


def _plot_confusion_matrix(y_true, y_pred, model_name):
    cm    = confusion_matrix(y_true, y_pred)
    disp  = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Failure", "Failure"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(f"{PLOTS_DIR}/cm_{fname}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/cm_{fname}.png")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
def plot_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [feature_names[i] for i in indices]
    sorted_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1],
                   color=["#4C72B0","#DD8452","#55A868","#C44E52"][::-1],
                   edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")

    for bar, imp in zip(bars, sorted_imp[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{imp:.3f}", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/feature_importance.png")
    print("\n  Feature Importance Rankings:")
    for feat, imp in zip(sorted_feat, sorted_imp):
        print(f"    {feat:20s}: {imp:.4f}")


# ─────────────────────────────────────────────
# 6. COMPARISON TABLE
# ─────────────────────────────────────────────
def compare_models(results):
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 40)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 40)
    print(df_res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df_res))
    width = 0.35
    ax.bar(x - width/2, df_res["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, df_res["roc_auc"],  width, label="ROC-AUC",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(df_res["model"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] Saved → {PLOTS_DIR}/model_comparison.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)

    # Train
    print("\n[...] Training Logistic Regression...")
    lr = train_logistic_regression(X_train_sc, y_train)

    print("[...] Training Random Forest...")
    rf = train_random_forest(X_train, y_train)     # RF doesn't need scaling

    # Evaluate
    results = []
    results.append(evaluate_model(lr, X_test_sc, y_test,
                                  "Logistic Regression", X_train_sc, y_train))
    results.append(evaluate_model(rf, X_test, y_test,
                                  "Random Forest", X_train, y_train))

    # Feature importance
    plot_feature_importance(rf, FEATURES)

    # Comparison
    compare_models(results)

    # Save best model (Random Forest typically wins)
    joblib.dump(rf, f"{MODEL_DIR}/random_forest_model.pkl")
    joblib.dump(lr, f"{MODEL_DIR}/logistic_regression_model.pkl")
    print("\n[✓] Models saved → model/random_forest_model.pkl")
    print("[✓] Models saved → model/logistic_regression_model.pkl")