"""
credit_scoring.py
=================
Production-quality credit scoring pipeline for bank loan default prediction.

The pipeline reads `credit_data.csv`, validates and cleans the dataset,
engineers domain-driven features, trains multiple models, evaluates them,
and saves model artifacts plus reports.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # non-interactive backend for automated report generation

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_PATH: Path = ROOT_DIR / "credit_data.csv"
DIR_FIGURES: Path = ROOT_DIR / "reports" / "figures"
DIR_METRICS: Path = ROOT_DIR / "reports" / "metrics"
DIR_MODELS: Path = ROOT_DIR / "models"
DIR_OUTPUTS: Path = ROOT_DIR / "outputs"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
TARGET_COL: str = "default"

REQUIRED_COLS: List[str] = [
    "income",
    "debt",
    "age",
    "loan_amount",
    "payment_history",
    "credit_score",
    "credit_utilization",
    "num_credit_lines",
    "late_payments",
    "employment_length",
    TARGET_COL,
]

PLOT_COLORS: Dict[str, str] = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "accent": "#16A34A",
    "neutral": "#6B7280",
}

logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def section(title: str) -> None:
    logger.info("\n" + "=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def step(message: str) -> None:
    logger.info(f"  ➤  {message}")


# =============================================================================
# PATHS
# =============================================================================

def create_directories() -> None:
    for folder in (DIR_FIGURES, DIR_METRICS, DIR_MODELS, DIR_OUTPUTS):
        folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place credit_data.csv in the project root."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV datasets are supported. Provide a .csv file.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Dataset file is empty: {path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"Failed to parse CSV file: {path}") from error

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {REQUIRED_COLS}."
        )

    if df[TARGET_COL].isna().any():
        raise ValueError("Target column contains missing values.")

    target_values = set(df[TARGET_COL].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target column values must be 0 or 1. Found: {sorted(target_values)}"
        )

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [col for col in REQUIRED_COLS if col not in numeric_cols]

    for col in non_numeric:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
            step(f"Converted column '{col}' to numeric type.")
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Column '{col}' must contain numeric values. "
                f"Found non-numeric values in the dataset."
            ) from error

    if df.empty:
        raise ValueError("Dataset is empty after schema validation.")

    return df


def report_data_quality(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    section(f"DATA QUALITY — {label}")
    missing = df.isnull().sum()
    duplicate_count = int(df.duplicated().sum())
    invalid_income = int((df["income"] <= 0).sum())
    invalid_age = int(((df["age"] < 18) | (df["age"] > 100)).sum())
    invalid_credit_score = int(
        ((df["credit_score"] < 300) | (df["credit_score"] > 850)).sum()
    )

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(missing.sum()),
        "missing_by_column": missing[missing > 0].to_dict(),
        "duplicate_rows": duplicate_count,
        "invalid_income": invalid_income,
        "invalid_age": invalid_age,
        "invalid_credit_score": invalid_credit_score,
        "default_ratio": df[TARGET_COL].mean(),
    }

    step(f"Rows                : {summary['rows']}")
    step(f"Columns             : {summary['columns']}")
    step(f"Missing values      : {summary['missing_values']}")
    if summary["missing_values"] > 0:
        step(f"Missing by column   : {summary['missing_by_column']}")
    step(f"Duplicate rows      : {summary['duplicate_rows']}")
    step(
        f"Invalid ranges      : income={invalid_income}, age={invalid_age}, credit_score={invalid_credit_score}"
    )
    step(f"Default ratio       : {summary['default_ratio']:.2%}")

    return summary


# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    section("EXPLORATORY DATA ANALYSIS")

    summary_path = DIR_METRICS / "eda_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("DATASET SUMMARY\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Shape: {df.shape}\n\n")
        handle.write("HEAD:\n")
        handle.write(df.head().to_string(index=False) + "\n\n")
        handle.write("DESCRIBE:\n")
        handle.write(df.describe(include="all").to_string() + "\n\n")
        handle.write("MISSING VALUES:\n")
        handle.write(df.isnull().sum().to_string() + "\n")

    step(f"EDA summary saved → {summary_path.relative_to(ROOT_DIR)}")

    feature_columns = [col for col in df.columns if col != TARGET_COL]
    _plot_target_distribution(df)
    _plot_correlation_matrix(df, feature_columns)
    _plot_feature_histograms(df, feature_columns)
    _plot_boxplots_by_target(df, feature_columns)


def _plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    bars = ax.bar(
        ["No Default (0)", "Default (1)"],
        counts.values,
        color=[PLOT_COLORS["primary"], PLOT_COLORS["secondary"]],
        width=0.5,
        edgecolor="white",
    )

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(5, count * 0.01),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title("Target Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "target_distribution.png", dpi=150)
    plt.close(fig)
    step("Plot saved → target_distribution.png")


def _plot_correlation_matrix(df: pd.DataFrame, feature_columns: List[str]) -> None:
    correlation = df[feature_columns + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    tick_labels = correlation.columns.to_list()
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_yticklabels(tick_labels, fontsize=8)

    for i in range(len(tick_labels)):
        for j in range(len(tick_labels)):
            ax.text(
                j,
                i,
                f"{correlation.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(correlation.iat[i, j]) > 0.5 else "black",
            )

    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    step("Plot saved → correlation_matrix.png")


def _plot_feature_histograms(df: pd.DataFrame, feature_columns: List[str]) -> None:
    n_columns = 3
    n_rows = (len(feature_columns) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, n_rows * 3))
    axes_flat = axes.flatten()

    for index, feature in enumerate(feature_columns):
        axes_flat[index].hist(df[feature].dropna(), bins=30, color=PLOT_COLORS["primary"], edgecolor="white", alpha=0.85)
        axes_flat[index].set_title(feature, fontsize=10, fontweight="bold")
        axes_flat[index].spines["top"].set_visible(False)
        axes_flat[index].spines["right"].set_visible(False)

    for empty_axis in axes_flat[len(feature_columns) :]:
        empty_axis.set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "feature_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    step("Plot saved → feature_histograms.png")


def _plot_boxplots_by_target(df: pd.DataFrame, feature_columns: List[str]) -> None:
    numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    n_columns = 3
    n_rows = (len(numeric_columns) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, n_rows * 3))
    axes_flat = axes.flatten()

    for index, feature in enumerate(numeric_columns):
        default_zero = df.loc[df[TARGET_COL] == 0, feature]
        default_one = df.loc[df[TARGET_COL] == 1, feature]
        box = axes_flat[index].boxplot(
            [default_zero, default_one],
            patch_artist=True,
            tick_labels=["No Default", "Default"],
            widths=0.5,
        )
        box["boxes"][0].set_facecolor(PLOT_COLORS["primary"])
        box["boxes"][1].set_facecolor(PLOT_COLORS["secondary"])
        for patch in box["boxes"]:
            patch.set_alpha(0.75)

        axes_flat[index].set_title(feature, fontsize=10, fontweight="bold")
        axes_flat[index].spines["top"].set_visible(False)
        axes_flat[index].spines["right"].set_visible(False)

    for empty_axis in axes_flat[len(numeric_columns) :]:
        empty_axis.set_visible(False)

    fig.suptitle("Feature Distribution by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "boxplots_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    step("Plot saved → boxplots_by_target.png")


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    section("DATA CLEANING")
    df = df.copy()

    before_count = len(df)
    df = df.drop_duplicates(ignore_index=True)
    duplicates_removed = before_count - len(df)
    step(f"Duplicates removed : {duplicates_removed}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        missing_count = int(df[column].isnull().sum())
        if missing_count > 0:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
            step(f"Imputed missing values for '{column}': {missing_count} rows")

    df["income"] = df["income"].clip(lower=1)
    df["age"] = df["age"].clip(lower=18, upper=100)
    df["credit_score"] = df["credit_score"].clip(lower=300, upper=850)
    df["credit_utilization"] = df["credit_utilization"].clip(lower=0.0, upper=1.0)
    df["num_credit_lines"] = df["num_credit_lines"].clip(lower=1)
    df["late_payments"] = df["late_payments"].clip(lower=0)
    df["payment_history"] = df["payment_history"].clip(lower=0, upper=10)
    df["employment_length"] = df["employment_length"].clip(lower=0)
    step("Invalid ranges clipped to valid banking limits")

    outlier_columns = [
        "income",
        "debt",
        "loan_amount",
        "credit_score",
        "credit_utilization",
    ]
    for column in outlier_columns:
        lower = df[column].quantile(0.01)
        upper = df[column].quantile(0.99)
        capped = df[column].clip(lower=lower, upper=upper)
        outlier_count = int((df[column] != capped).sum())
        df[column] = capped
        if outlier_count > 0:
            step(f"Outliers capped for '{column}': {outlier_count} rows")

    report_data_quality(df, "After Cleaning")
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    section("FEATURE ENGINEERING")
    df = df.copy()

    df["debt_to_income"] = df["debt"] / (df["income"] + 1)
    step("Created feature: debt_to_income")

    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    step("Created feature: loan_to_income")

    df["income_per_credit_line"] = df["income"] / (df["num_credit_lines"] + 1)
    step("Created feature: income_per_credit_line")

    df["late_payment_ratio"] = df["late_payments"] / (df["num_credit_lines"] + 1)
    step("Created feature: late_payment_ratio")

    df["credit_utilization"] = df["credit_utilization"].clip(lower=0.0, upper=1.0)
    df["utilization_risk"] = pd.cut(
        df["credit_utilization"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=[0, 1, 2],
        include_lowest=True,
        right=True,
    ).astype(int)
    step("Created feature: utilization_risk")

    normalized_score = (df["credit_score"] - 300) / 550
    df["risk_score_index"] = (
        df["credit_utilization"] * 0.35
        + df["late_payment_ratio"] * 0.30
        + (1.0 - normalized_score) * 0.35
    )
    step("Created feature: risk_score_index")

    log_message = f"Total features after engineering: {len(df.columns) - 1}"
    step(log_message)
    return df


# =============================================================================
# SPLIT & PREPROCESS
# =============================================================================

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    section("DATA SPLITTING")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    if y.nunique() < 2:
        raise ValueError("Target column must contain both classes 0 and 1 for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    step(f"Train samples: {len(X_train)}")
    step(f"Test samples : {len(X_test)}")
    return X_train, X_test, y_train, y_test


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    section("FEATURE SCALING")
    scaler = scaler or StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    step("StandardScaler fit on training data only")
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_models(X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
    section("MODEL TRAINING")

    logistic_regression = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    logistic_regression.fit(X_train, y_train)
    step("Logistic Regression trained")

    decision_tree = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    decision_tree.fit(X_train, y_train)
    step("Decision Tree trained")

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)
    step("Random Forest trained")

    return {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def _get_probability(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise AttributeError("Model does not support probability or decision function.")


def evaluate_model(name: str, model: Any, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = _get_probability(model, X_test)
    cm = confusion_matrix(y_test, y_pred)

    auc_value = roc_auc_score(y_test, y_proba)
    metrics = {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc_value,
        "confusion_matrix": cm,
    }

    logger.info(f"\n  ┌── {name} {'─' * max(0, 36 - len(name))}┐")
    logger.info(f"  │ Accuracy   : {metrics['accuracy']:.4f}")
    logger.info(f"  │ Precision  : {metrics['precision']:.4f}")
    logger.info(f"  │ Recall     : {metrics['recall']:.4f}")
    logger.info(f"  │ F1-Score   : {metrics['f1_score']:.4f}")
    logger.info(f"  │ ROC-AUC    : {metrics['roc_auc']:.4f}")
    logger.info(
        f"  │ Confusion Matrix → TN={cm[0,0]:>4}  FP={cm[0,1]:>4}  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}"
    )
    logger.info(f"  └{'─' * 52}┘")

    report_path = DIR_METRICS / f"{name.replace(' ', '_').lower()}_report.txt"
    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write(f"Classification Report — {name}\n")
        report_file.write("=" * 60 + "\n")
        report_file.write(
            classification_report(
                y_test,
                y_pred,
                target_names=["No Default", "Default"],
                zero_division=0,
            )
        )
    step(f"Classification report saved → {report_path.relative_to(ROOT_DIR)}")

    return metrics


def evaluate_all_models(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series) -> List[Dict[str, Any]]:
    section("MODEL EVALUATION")
    return [evaluate_model(name, model, X_test, y_test) for name, model in models.items()]


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    section("MODEL COMPARISON")
    sorted_results = sorted(results, key=lambda result: result["roc_auc"], reverse=True)

    logger.info("\n  Rank  Model                  Accuracy  Precision  Recall  F1     ROC-AUC")
    logger.info("  " + "─" * 70)
    for rank, result in enumerate(sorted_results, start=1):
        best_label = " ← BEST" if rank == 1 else ""
        logger.info(
            f"  {rank:<4} {result['name']:<22} "
            f"{result['accuracy']:.4f}   {result['precision']:.4f}   "
            f"{result['recall']:.4f}  {result['f1_score']:.4f}  "
            f"{result['roc_auc']:.4f}{best_label}"
        )

    comparison = pd.DataFrame(
        [
            {
                "name": result["name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
            }
            for result in sorted_results
        ]
    )
    comparison_path = DIR_METRICS / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    step(f"Model comparison saved → {comparison_path.relative_to(ROOT_DIR)}")
    step(f"Best model selected → {sorted_results[0]['name']}")
    return sorted_results[0]


# =============================================================================
# FIGURES
# =============================================================================

def plot_curves(results: List[Dict[str, Any]], y_test: pd.Series) -> None:
    section("ROC / Precision-Recall Curves")
    colors = [PLOT_COLORS["primary"], PLOT_COLORS["secondary"], PLOT_COLORS["accent"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, result in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        precision, recall, _ = precision_recall_curve(y_test, result["y_proba"])
        axes[0].plot(fpr, tpr, color=colors[idx], lw=2, label=f"{result['name']} (AUC={result['roc_auc']:.3f})")
        axes[1].plot(recall, precision, color=colors[idx], lw=2, label=result["name"])

    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "roc_pr_curves.png", dpi=150)
    plt.close(fig)
    step("Plot saved → roc_pr_curves.png")


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(result: Dict[str, Any], feature_names: List[str]) -> List[str]:
    section("FEATURE IMPORTANCE")
    model = result["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance Score"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        label = "Absolute Coefficient"
    else:
        step("Feature importance is not available for this model.")
        return []

    importance_df = (
        pd.DataFrame({"feature": feature_names, "value": importances})
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    importance_path = DIR_METRICS / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    step(f"Feature importance saved → {importance_path.relative_to(ROOT_DIR)}")

    top10 = importance_df.head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [PLOT_COLORS["primary"] if i < 5 else PLOT_COLORS["neutral"] for i in range(len(top10))]
    ax.barh(top10["feature"][::-1], top10["value"][::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel(label)
    ax.set_title(f"Top 10 Features — {result['name']}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in ax.patches:
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(DIR_FIGURES / "feature_importance.png", dpi=150)
    plt.close(fig)
    step("Plot saved → feature_importance.png")

    step(f"Top 10 features printed for {result['name']}")
    return top10["feature"].tolist()


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_artifacts(
    result: Dict[str, Any],
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
) -> None:
    section("SAVING ARTIFACTS")

    model_path = DIR_MODELS / "credit_scoring_model.pkl"
    scaler_path = DIR_MODELS / "scaler.pkl"
    joblib.dump(result["model"], model_path)
    joblib.dump(scaler, scaler_path)
    step(f"Model saved → {model_path.relative_to(ROOT_DIR)}")
    step(f"Scaler saved → {scaler_path.relative_to(ROOT_DIR)}")

    predictions = X_test.copy()
    predictions["actual_default"] = y_test.reset_index(drop=True)
    predictions["predicted_default"] = result["y_pred"]
    predictions["default_probability"] = np.round(result["y_proba"], 4)
    predictions_path = DIR_OUTPUTS / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    step(f"Predictions saved → {predictions_path.relative_to(ROOT_DIR)}")


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    result: Dict[str, Any],
    total_rows: int,
    train_rows: int,
    test_rows: int,
    top_features: List[str],
) -> None:
    section("FINAL EXECUTION SUMMARY")

    logger.info(f"  Best Model             : {result['name']}")
    logger.info(f"  Accuracy               : {result['accuracy']:.4f}")
    logger.info(f"  Precision              : {result['precision']:.4f}")
    logger.info(f"  Recall                 : {result['recall']:.4f}")
    logger.info(f"  F1 Score               : {result['f1_score']:.4f}")
    logger.info(f"  ROC-AUC                : {result['roc_auc']:.4f}")
    logger.info(f"  Dataset size           : {total_rows}")
    logger.info(f"  Training samples       : {train_rows}")
    logger.info(f"  Testing samples        : {test_rows}")

    if top_features:
        logger.info("\n  Top 5 Features:")
        for rank, feature in enumerate(top_features[:5], start=1):
            logger.info(f"    {rank}. {feature}")

    logger.info("\n  Saved artifacts:")
    logger.info(f"    models/credit_scoring_model.pkl")
    logger.info(f"    models/scaler.pkl")
    logger.info(f"    outputs/predictions.csv")
    logger.info(f"    reports/metrics/model_comparison.csv")
    logger.info(f"    reports/metrics/feature_importance.csv")
    logger.info(f"    reports/metrics/eda_summary.txt")
    logger.info(f"    reports/figures/*.png")
    logger.info("\n  Pipeline completed successfully.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    configure_logging()
    create_directories()

    section("LOADING DATA")
    df_raw = load_dataset(DATA_PATH)
    step(f"Loaded dataset: {len(df_raw)} rows x {len(df_raw.columns)} columns")

    df_raw = validate_schema(df_raw)
    report_data_quality(df_raw, "Raw Dataset")
    run_eda(df_raw)

    df_clean = clean_data(df_raw)
    df_clean = validate_schema(df_clean)
    report_data_quality(df_clean, "Cleaned Dataset")

    df_engineered = engineer_features(df_clean)
    df_engineered = validate_schema(df_engineered)
    report_data_quality(df_engineered, "Engineered Dataset")

    X_train, X_test, y_train, y_test = split_data(df_engineered)
    X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

    models = train_models(X_train_scaled, y_train)
    results = evaluate_all_models(models, X_test_scaled, y_test)
    plot_curves(results, y_test)
    best_result = compare_models(results)
    top_features = plot_feature_importance(best_result, X_train.columns.tolist())
    save_artifacts(best_result, scaler, X_test.reset_index(drop=True), y_test.reset_index(drop=True), X_train.columns.tolist())
    print_summary(best_result, len(df_raw), len(X_train), len(X_test), top_features)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logger.exception("A fatal error occurred while running the credit scoring pipeline.")
        sys.exit(1)
