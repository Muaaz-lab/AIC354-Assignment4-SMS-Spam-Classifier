"""Train and compare six TF-IDF text classifiers for Assignment 4.

Run from this folder with:
    python train_models.py

The vectorizer is fitted only on the training portion for model comparison.
After choosing the best algorithm by macro-F1, the saved deployment pipeline
is refitted on the complete dataset.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "SMSSpamCollection.tsv"
MODEL_PATH = ROOT / "models" / "best_sms_spam_model.joblib"
RESULTS_DIR = ROOT / "results"
LABELS = ["ham", "spam"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_vectorizer() -> TfidfVectorizer:
    """Create the text representation used for every candidate classifier."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )


def make_models() -> dict[str, object]:
    """Return six supervised classifiers with deterministic settings."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=250,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ),
    }


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    if len(data) != 5_574:
        raise ValueError(f"Expected all 5,574 UCI messages; found {len(data):,}.")
    if data[["label", "message"]].isna().any().any():
        raise ValueError("The dataset contains a missing label or message.")
    if set(data["label"].unique()) != set(LABELS):
        raise ValueError("Expected exactly the ham and spam labels.")
    return data


def save_plots(data: pd.DataFrame, comparison: pd.DataFrame, y_test, y_pred, best_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = data["label"].value_counts().reindex(LABELS)
    bars = ax.bar(order.index.str.title(), order.values, color=["#4C78A8", "#F58518"])
    ax.set(title="SMS message classes", ylabel="Number of messages", xlabel="Label")
    ax.bar_label(bars, fmt="%d", padding=4)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plot_data = comparison.sort_values("Macro F1", ascending=True)
    positions = range(len(plot_data))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(list(positions), plot_data["Macro F1"], color="#4C78A8", label="Macro F1")
    ax.scatter(plot_data["Spam recall"], positions, color="#F58518", s=42, label="Spam recall", zorder=3)
    ax.set_yticks(list(positions), plot_data["Model"])
    ax.set(xlim=(0, 1.03), xlabel="Score", title="Held-out model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar=False,
        ax=ax,
    )
    ax.set(xlabel="Predicted label", ylabel="Actual label", title=f"{best_name}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "best_model_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> dict[str, object]:
    """Compare candidate algorithms, save results, and fit the deployment model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    vectorizer = make_vectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    candidate_models = make_models()
    fitted_models: dict[str, object] = {}
    prediction_by_model: dict[str, pd.Series] = {}
    rows: list[dict[str, float | str]] = []

    for name, estimator in candidate_models.items():
        started = time.perf_counter()
        estimator.fit(x_train_tfidf, y_train)
        fit_seconds = time.perf_counter() - started
        prediction = estimator.predict(x_test_tfidf)
        fitted_models[name] = estimator
        prediction_by_model[name] = pd.Series(prediction, index=y_test.index)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Balanced accuracy": balanced_accuracy_score(y_test, prediction),
                "Macro F1": f1_score(y_test, prediction, average="macro"),
                "Spam precision": precision_score(y_test, prediction, pos_label="spam", zero_division=0),
                "Spam recall": recall_score(y_test, prediction, pos_label="spam", zero_division=0),
                "Spam F1": f1_score(y_test, prediction, pos_label="spam", zero_division=0),
                "Fit seconds": fit_seconds,
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["Macro F1", "Spam recall"], ascending=False, ignore_index=True
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False, float_format="%.6f")
    best_name = str(comparison.loc[0, "Model"])
    best_prediction = prediction_by_model[best_name].loc[y_test.index].to_numpy()
    report = classification_report(
        y_test, best_prediction, labels=LABELS, output_dict=True, zero_division=0
    )
    save_plots(data, comparison, y_test, best_prediction, best_name)

    # Train the deployment artifact on all labeled messages, retaining the exact
    # preprocessing steps inside the saved pipeline.
    final_pipeline = Pipeline(
        [("tfidf", make_vectorizer()), ("classifier", clone(candidate_models[best_name]))]
    )
    final_pipeline.fit(data["message"], data["label"])
    joblib.dump(final_pipeline, MODEL_PATH, compress=3)

    class_counts = data["label"].value_counts().reindex(LABELS).astype(int).to_dict()
    summary = {
        "dataset": "UCI SMS Spam Collection",
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "rows": int(len(data)),
        "class_counts": class_counts,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "vocabulary_size": int(x_train_tfidf.shape[1]),
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 30_000,
            "sublinear_tf": True,
            "stop_words": None,
        },
        "best_model_by": "macro F1, with spam recall as the tie-breaker",
        "best_model": best_name,
        "best_test_metrics": comparison.iloc[0].to_dict(),
        "best_test_classification_report": report,
        "saved_pipeline": str(MODEL_PATH.relative_to(ROOT)),
        "scikit_learn_version": sklearn.__version__,
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Dataset: {len(data):,} messages ({class_counts['ham']:,} ham, {class_counts['spam']:,} spam)")
    print(f"Split: {len(x_train):,} train / {len(x_test):,} test, stratified, random_state={RANDOM_STATE}")
    print(f"TF-IDF vocabulary: {x_train_tfidf.shape[1]:,} terms")
    print(f"Best model by macro F1: {best_name}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTest classification report for the selected model:")
    print(classification_report(y_test, best_prediction, labels=LABELS, zero_division=0))
    print(f"Saved deployment pipeline: {MODEL_PATH}")
    return {
        "data": data,
        "comparison": comparison,
        "best_model_name": best_name,
        "y_test": y_test,
        "y_pred": best_prediction,
        "summary": summary,
    }


if __name__ == "__main__":
    run_experiment()

