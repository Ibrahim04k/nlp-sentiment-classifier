"""
Model Training and Benchmarking Pipeline for Sentiment Analysis.
Benchmarks Logistic Regression, Calibrated LinearSVC, and Multinomial Naive Bayes.
Evaluates metrics (Accuracy, Macro F1, ROC-AUC) and exports the winning pipeline.
"""

import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Add project root to path for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CSV_PATH = os.path.join(DATA_DIR, "reviews.csv")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
PIPELINE_PATH = os.path.join(MODELS_DIR, "sentiment_tfidf_pipeline.pkl")

DATASET_URL = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv"

def ensure_dataset(data_path: str = CSV_PATH, sample_size: int = 30000) -> pd.DataFrame:
    """
    Automated dataset loader:
    1. Checks if data/reviews.csv exists locally.
    2. If missing, automatically downloads the IMDB 50k sentiment benchmark dataset.
    3. Samples to a balanced subset for fast local execution while preserving high accuracy.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    if os.path.exists(data_path):
        print(f"[Dataset] Loading existing dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print(f"[Dataset] Dataset not found locally. Downloading IMDB benchmark from public repository...")
        print(f"          URL: {DATASET_URL}")
        try:
            # Download with progress report
            def _report_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / (total_size + 1e-5))
                if percent % 25 == 0 and percent <= 100:
                    sys.stdout.write(f"\r          Progress: {min(percent, 100)}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(DATASET_URL, data_path, reporthook=_report_hook)
            print("\n[Dataset] Download complete! Saved to:", data_path)
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"[Dataset] Download failed: {e}. Generating synthetic fallback dataset for tests...")
            df = _generate_fallback_dataset()
            df.to_csv(data_path, index=False)

    # Standardize column names
    col_map = {col: col.lower().strip() for col in df.columns}
    df = df.rename(columns=col_map)
    
    text_col = "review" if "review" in df.columns else df.columns[0]
    sentiment_col = "sentiment" if "sentiment" in df.columns else df.columns[1]
    
    df = df[[text_col, sentiment_col]].dropna()
    df.columns = ["review", "sentiment"]
    
    # Map sentiment to binary (1 = Positive, 0 = Negative)
    if df["sentiment"].dtype == object or isinstance(df["sentiment"].iloc[0], str):
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        df["label"] = df["sentiment"].map({"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0})
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = df["sentiment"].astype(int)

    # Balanced sampling for fast local training
    if sample_size and len(df) > sample_size:
        print(f"[Dataset] Subsampling {sample_size:,} balanced records for fast, optimal local training...")
        pos_df = df[df["label"] == 1].sample(n=sample_size // 2, random_state=42)
        neg_df = df[df["label"] == 0].sample(n=sample_size // 2, random_state=42)
        df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[Dataset] Final dataset shape: {df.shape[0]:,} reviews. Class balance: {dict(df['label'].value_counts())}")
    return df

def _generate_fallback_dataset() -> pd.DataFrame:
    """Fallback generator in case network is completely unreachable."""
    pos = [
        "This is an absolute masterpiece! Exceptional acting and brilliant direction.",
        "One of the greatest films ever made. Truly heartwarming and deeply moving.",
        "Loved everything about this product. High quality, excellent performance, highly recommend!",
        "Super fast shipping, works perfectly right out of the box, five stars!",
        "Fantastic experience, superb customer support and very reliable product."
    ] * 200
    neg = [
        "Terrible waste of time. Boring plot, awful dialogue and terrible execution.",
        "Completely disappointed. Broke after two days and customer support was useless.",
        "Do not buy this! It does not work as advertised and quality is very cheap.",
        "Worst experience ever. Slow, glitchy, and constantly crashed.",
        "Total garbage, not worth a single penny. Avoid at all costs."
    ] * 200
    df = pd.DataFrame({
        "review": pos + neg,
        "sentiment": ["positive"] * len(pos) + ["negative"] * len(neg)
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def train_and_benchmark(sample_size: int = 30000):
    """
    Executes the complete training and benchmarking cycle across:
    1. Logistic Regression
    2. LinearSVC (Calibrated with CalibratedClassifierCV)
    3. Multinomial Naive Bayes
    """
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 75)
    print("  CUSTOMER REVIEW SENTIMENT ANALYSIS - NLP TRAINING & BENCHMARKING")
    print("=" * 75)
    
    df = ensure_dataset(sample_size=sample_size)
    
    X = df["review"]
    y = df["label"]
    
    print("\n[Splitting] Creating stratified Train/Test splits (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"            Training samples: {len(X_train):,}")
    print(f"            Testing samples : {len(X_test):,}")

    # Shared TF-IDF Vectorizer configuration
    # Unigrams + Bigrams, sublinear TF scaling, max 25,000 features
    tfidf_config = {
        "preprocessor": clean_text,
        "ngram_range": (1, 2),
        "max_features": 25000,
        "sublinear_tf": True
    }

    # Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        ),
        "Calibrated LinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, dual="auto"),
            method="sigmoid",
            cv=3
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0
        )
    }

    results = {}
    fitted_pipelines = {}

    print("\n" + "-" * 75)
    print(f"{'Model':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'Macro F1':<9} | {'ROC-AUC':<9}")
    print("-" * 75)

    for name, clf in models.items():
        t0 = time.time()
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("classifier", clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Probabilities for ROC-AUC
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_proba = pipeline.decision_function(X_test)
        else:
            y_proba = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        elapsed = time.time() - t0

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "macro_f1": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "training_time_seconds": round(float(elapsed), 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)
        }
        fitted_pipelines[name] = pipeline

        print(f"{name:<26} | {acc*100:>7.2f}% | {prec*100:>7.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {roc*100:>7.2f}%  ({elapsed:.1f}s)")

    print("-" * 75)

    # Identify Best Model by Macro F1
    best_model_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n[Winner] Best Performing Model: >>> {best_model_name} <<<")
    print(f"         Accuracy : {best_metrics['accuracy'] * 100:.2f}%")
    print(f"         Macro F1 : {best_metrics['macro_f1'] * 100:.2f}%")
    print(f"         ROC-AUC  : {best_metrics['roc_auc'] * 100:.2f}%")

    # Save Pipeline and Metrics
    print(f"\n[Saving] Exporting best pipeline to: {PIPELINE_PATH}")
    joblib.dump(best_pipeline, PIPELINE_PATH, compress=3)

    metrics_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_model_name,
        "sample_size": len(df),
        "test_size": len(X_test),
        "benchmark": results
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"[Saving] Saved benchmark metrics JSON to: {METRICS_PATH}")

    # Plot & Save Visuals
    _save_evaluation_plots(results, best_model_name, best_metrics)

    total_time = time.time() - start_time
    print(f"\n[Finished] Total pipeline execution completed in {total_time:.1f} seconds.")
    print("=" * 75)
    return results

def _save_evaluation_plots(results: dict, best_name: str, best_metrics: dict):
    """Generate high-resolution benchmark plots for documentation."""
    try:
        # 1. Confusion Matrix for Best Model
        cm = np.array(best_metrics["confusion_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            annot_kws={"size": 14, "weight": "bold"}
        )
        plt.title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("True Label", fontsize=11)
        plt.xlabel("Predicted Label", fontsize=11)
        plt.tight_layout()
        cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[Visuals] Saved Confusion Matrix plot to: {cm_path}")

        # 2. Benchmark Comparison Bar Chart
        models = list(results.keys())
        accs = [results[m]["accuracy"] * 100 for m in models]
        f1s = [results[m]["macro_f1"] * 100 for m in models]
        rocs = [results[m]["roc_auc"] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.25

        plt.figure(figsize=(9, 5))
        plt.bar(x - width, accs, width, label="Accuracy (%)", color="#3b82f6")
        plt.bar(x, f1s, width, label="Macro F1 (%)", color="#10b981")
        plt.bar(x + width, rocs, width, label="ROC-AUC (%)", color="#8b5cf6")

        plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
        plt.title("NLP Model Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
        plt.xticks(x, models, fontsize=10, fontweight="bold")
        plt.ylim(70, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.legend(frameon=True, facecolor="white", loc="lower right")
        plt.tight_layout()
        bench_path = os.path.join(MODELS_DIR, "model_comparison.png")
        plt.savefig(bench_path, dpi=200)
        plt.close()
        print(f"[Visuals] Saved Model Comparison plot to: {bench_path}")

    except Exception as e:
        print(f"[Warning] Could not save evaluation plots: {e}")

if __name__ == "__main__":
    train_and_benchmark(sample_size=30000)
