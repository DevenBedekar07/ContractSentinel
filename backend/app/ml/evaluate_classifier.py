
"""
ContractSentinel: Evaluate category-specific TF-IDF + Linear SVM models.

This initial evaluation uses held-out CUAD test passages with exactly
one label, matching the single-label training setup.

Because CUAD annotations are not exhaustive negative labels, the
reported metrics should be interpreted as preliminary estimates.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TEST_FILE = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "processed"
    / "cuad"
    / "test_spans.jsonl"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "ml"
    / "artifacts"
)

MODEL_FILE = ARTIFACT_DIR / "clause_classifiers.joblib"

OUTPUT_FILE = (
    ARTIFACT_DIR
    / "evaluation_results.json"
)


# ---------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------

def load_test_data(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test file not found: {path}\n"
            "Run prepare_dataset.py first."
        )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            text = record.get("text", "").strip()
            labels = record.get("labels", [])

            if not text or not isinstance(labels, list) or not labels:
                continue

            record["text"] = text
            record["labels"] = sorted(set(labels))
            records.append(record)

    return records


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "positive_test_examples": int(sum(y_true)),
        "negative_test_examples": int(len(y_true) - sum(y_true)),
    }


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("CONTRACTSENTINEL - MODEL EVALUATION")
    print("=" * 72)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained models not found: {MODEL_FILE}\n"
            "Run train_classifier.py first."
        )

    models = joblib.load(MODEL_FILE)
    all_records = load_test_data(TEST_FILE)

    # Use the same label scope as the initial training experiment.
    test_records = [
        record
        for record in all_records
        if len(record["labels"]) == 1
    ]

    excluded_multilabel = len(all_records) - len(test_records)

    if not test_records:
        raise RuntimeError("No single-label test passages found.")

    print(f"Test passages loaded: {len(all_records)}")
    print(f"Single-label test passages: {len(test_records)}")
    print(f"Multi-label passages excluded: {excluded_multilabel}")
    print(f"Models loaded: {len(models)}")
    print()

    results = {
        "project": "ContractSentinel",
        "evaluation_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_type": "TF-IDF + LinearSVC",
        "test_file": str(TEST_FILE),
        "evaluation_scope": (
            "Single-label annotated CUAD passages only. "
            "Not full-contract clause-detection evaluation."
        ),
        "negative_label_warning": (
            "Metrics are preliminary because CUAD annotations may not "
            "provide exhaustive verified negative labels."
        ),
        "total_test_passages": len(all_records),
        "single_label_test_passages": len(test_records),
        "excluded_multilabel_passages": excluded_multilabel,
        "models_evaluated": 0,
        "categories": {},
    }

    # A single-label passage has one known category.
    texts = [record["text"] for record in test_records]
    true_categories = [
        record["labels"][0] for record in test_records
    ]

    for index, (category, model) in enumerate(
        sorted(models.items()),
        start=1,
    ):
        print(f"[{index}/{len(models)}] Evaluating: {category}")

        y_true = [
            1 if label == category else 0
            for label in true_categories
        ]

        y_pred = model.predict(texts).astype(int).tolist()

        metrics = calculate_metrics(y_true, y_pred)

        results["categories"][category] = {
            "status": "evaluated",
            **metrics,
        }

        print(
            f"  Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    results["models_evaluated"] = len(models)

    evaluated = list(results["categories"].values())

    if evaluated:
        results["macro_average"] = {
            "precision": round(
                sum(item["precision"] for item in evaluated)
                / len(evaluated),
                4,
            ),
            "recall": round(
                sum(item["recall"] for item in evaluated)
                / len(evaluated),
                4,
            ),
            "f1_score": round(
                sum(item["f1_score"] for item in evaluated)
                / len(evaluated),
                4,
            ),
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print()
    print("=" * 72)
    print("EVALUATION COMPLETED")
    print("=" * 72)
    print(f"Models evaluated: {results['models_evaluated']}")

    if "macro_average" in results:
        print(
            "Macro precision: "
            f"{results['macro_average']['precision']:.4f}"
        )
        print(
            "Macro recall: "
            f"{results['macro_average']['recall']:.4f}"
        )
        print(
            "Macro F1-score: "
            f"{results['macro_average']['f1_score']:.4f}"
        )

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")
    print(
        "\nReminder: these are preliminary annotated-passage metrics, "
        "not proof of reliable full-contract detection."
    )


if __name__ == "__main__":
    main()
