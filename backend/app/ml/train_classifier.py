
"""
ContractSentinel: Train category-specific TF-IDF + Linear SVM classifiers.

Input:
    backend/data/processed/cuad/train_spans.jsonl

Output:
    backend/app/ml/artifacts/clause_classifiers.joblib
    backend/app/ml/artifacts/label_metadata.json

Important:
    CUAD contains positive annotations, not exhaustive verified negatives.
    This script uses single-label annotated spans as positives and spans
    explicitly annotated for other categories as provisional negatives.
    Multi-label spans are excluded from binary training to reduce ambiguity.

    These negatives are weak labels, not confirmed legal negatives.
"""

import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "processed"
    / "cuad"
    / "train_spans.jsonl"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "ml"
    / "artifacts"
)

MODEL_FILE = ARTIFACT_DIR / "clause_classifiers.joblib"
METADATA_FILE = ARTIFACT_DIR / "label_metadata.json"


# ---------------------------------------------------------
# TRAINING CONFIGURATION
# ---------------------------------------------------------

MIN_POSITIVE_EXAMPLES = 20
MIN_NEGATIVE_EXAMPLES = 20

TFIDF_CONFIG = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "max_features": 100_000,
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.98,
}

SVM_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 5000,
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load one JSON object per line."""
    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}\n"
            "Run prepare_dataset.py first."
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            text = record.get("text", "").strip()
            labels = record.get("labels", [])

            if not text or not isinstance(labels, list) or not labels:
                continue

            record["text"] = text
            record["labels"] = sorted(set(labels))
            records.append(record)

    if not records:
        raise ValueError("No valid training records were found.")

    return records


# ---------------------------------------------------------
# BUILD CATEGORY-SPECIFIC DATA
# ---------------------------------------------------------

def build_category_examples(
    records: list[dict],
) -> tuple[list[str], dict[str, dict[str, list]]]:
    """
    Build binary training examples for every category.

    Single-label span:
        Its label is positive.
        Other categories treat it as a provisional negative.

    Multi-label span:
        Excluded from binary training because its labels overlap.

    Note:
        The negative examples remain weakly labeled. CUAD's annotations
        do not prove that every other category is absent from a span.
    """

    category_counts = Counter()

    for record in records:
        category_counts.update(record["labels"])

    categories = sorted(category_counts.keys())

    examples = {
        category: {"texts": [], "labels": []}
        for category in categories
    }

    excluded_multilabel = 0

    for record in records:
        labels = record["labels"]
        text = record["text"]

        # Exclude overlapping annotations from binary training.
        if len(labels) != 1:
            excluded_multilabel += 1
            continue

        actual_label = labels[0]

        for category in categories:
            examples[category]["texts"].append(text)
            examples[category]["labels"].append(
                1 if actual_label == category else 0
            )

    print(f"Unique categories: {len(categories)}")
    print(f"Records loaded: {len(records)}")
    print(f"Multi-label records excluded: {excluded_multilabel}")
    print()

    return categories, examples


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

def train_models(
    categories: list[str],
    examples: dict[str, dict[str, list]],
) -> tuple[dict, dict]:
    """Train and collect a separate binary SVM for each category."""

    models = {}
    metadata = {
        "project": "ContractSentinel",
        "model_type": "TF-IDF + LinearSVC",
        "training_file": str(TRAIN_FILE),
        "tfidf_config": TFIDF_CONFIG,
        "svm_config": SVM_CONFIG,
        "negative_label_warning": (
            "Negative examples are provisional: they are single-label "
            "CUAD spans assigned to other categories, not exhaustively "
            "verified negative passages."
        ),
        "excluded_multilabel_spans": True,
        "categories": {},
    }

    for index, category in enumerate(categories, start=1):
        texts = examples[category]["texts"]
        labels = examples[category]["labels"]

        positive_count = sum(labels)
        negative_count = len(labels) - positive_count

        print(f"[{index}/{len(categories)}] {category}")
        print(f"  Positive examples: {positive_count}")
        print(f"  Provisional negatives: {negative_count}")

        if positive_count < MIN_POSITIVE_EXAMPLES:
            print("  SKIPPED: too few positive examples.")
            metadata["categories"][category] = {
                "status": "skipped",
                "reason": "too_few_positive_examples",
                "positive_examples": positive_count,
                "negative_examples": negative_count,
            }
            print()
            continue

        if negative_count < MIN_NEGATIVE_EXAMPLES:
            print("  SKIPPED: too few negative examples.")
            metadata["categories"][category] = {
                "status": "skipped",
                "reason": "too_few_negative_examples",
                "positive_examples": positive_count,
                "negative_examples": negative_count,
            }
            print()
            continue

        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(**TFIDF_CONFIG),
                ),
                (
                    "classifier",
                    LinearSVC(**SVM_CONFIG),
                ),
            ]
        )

        pipeline.fit(texts, labels)

        models[category] = pipeline

        metadata["categories"][category] = {
            "status": "trained",
            "positive_examples": positive_count,
            "negative_examples": negative_count,
            "total_examples": len(labels),
        }

        print("  Status: TRAINED")
        print()

    if not models:
        raise RuntimeError(
            "No models were trained. Check the dataset and thresholds."
        )

    return models, metadata


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("CONTRACTSENTINEL - TF-IDF + LINEAR SVM TRAINING")
    print("=" * 68)
    print()

    records = load_jsonl(TRAIN_FILE)

    categories, examples = build_category_examples(records)

    models, metadata = train_models(categories, examples)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(models, MODEL_FILE)

    metadata["trained_model_count"] = len(models)
    metadata["category_count"] = len(categories)

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    trained_categories = [
        category
        for category, details in metadata["categories"].items()
        if details["status"] == "trained"
    ]

    skipped_categories = [
        category
        for category, details in metadata["categories"].items()
        if details["status"] == "skipped"
    ]

    print("=" * 68)
    print("TRAINING COMPLETED")
    print("=" * 68)
    print(f"Models trained: {len(models)}")
    print(f"Categories found: {len(categories)}")
    print(f"Categories skipped: {len(skipped_categories)}")
    print()
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print()
    print("Trained categories:")

    for category in trained_categories:
        print(f"  - {category}")

    if skipped_categories:
        print()
        print("Skipped categories:")

        for category in skipped_categories:
            details = metadata["categories"][category]
            print(f"  - {category}: {details['reason']}")

    print()
    print("IMPORTANT:")
    print(
        "These models classify annotated passages. They have not yet "
        "been evaluated or integrated into the application."
    )


if __name__ == "__main__":
    main()
