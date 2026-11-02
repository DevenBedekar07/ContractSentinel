import json
import re
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CUAD_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "raw"
    / "cuad"
    / "extracted"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
    / "processed"
    / "cuad"
)

TRAIN_FILE = CUAD_DIR / "train_separate_questions.json"
TEST_FILE = CUAD_DIR / "test.json"


# ---------------------------------------------------------
# CATEGORY EXTRACTION
# ---------------------------------------------------------

CATEGORY_PATTERN = re.compile(
    r'related to "([^"]+)"'
)


def extract_category(question: str) -> str | None:
    """
    Extract the CUAD category from an annotation question.

    Example:
    Highlight the parts of this contract related to
    "Governing Law" ...

    Returns:
        Governing Law
    """

    match = CATEGORY_PATTERN.search(question)

    if match:
        return match.group(1).strip()

    return None


# ---------------------------------------------------------
# LOAD JSON
# ---------------------------------------------------------

def load_dataset(path: Path) -> dict:
    """Load a CUAD JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# PREPARE ANNOTATED SPANS
# ---------------------------------------------------------

def prepare_spans(path: Path, split_name: str) -> dict:
    """
    Extract annotated answer spans and merge category
    labels when the same span has multiple annotations.

    This produces annotated-span classification data.

    It does not generate arbitrary negative examples
    from unannotated contract text.
    """

    dataset = load_dataset(path)

    contracts = dataset.get("data", [])

    # Key:
    # (contract title, start offset, end offset, text)
    #
    # Value:
    # Record containing text and its category labels.

    spans = {}

    stats = Counter()
    category_counts = Counter()
    contract_titles = set()

    for contract in contracts:
        title = contract.get("title", "")

        if not title:
            stats["contracts_without_title"] += 1

        contract_titles.add(title)

        for paragraph in contract.get("paragraphs", []):
            context = paragraph.get("context", "")

            for qa in paragraph.get("qas", []):
                stats["annotation_questions"] += 1

                question = qa.get("question", "")
                category = extract_category(question)

                if category is None:
                    stats["questions_without_category"] += 1
                    continue

                answers = qa.get("answers", [])

                if not answers:
                    stats["questions_without_answers"] += 1
                    continue

                for answer in answers:
                    text = answer.get("text", "")
                    start = answer.get("answer_start", -1)

                    if not text.strip():
                        stats["empty_answer_texts"] += 1
                        continue

                    if not isinstance(start, int) or start < 0:
                        stats["invalid_start_offsets"] += 1
                        continue

                    end = start + len(text)

                    # Validate the annotated character offsets.
                    if context[start:end] != text:
                        stats["offset_mismatches"] += 1
                        continue

                    key = (title, start, end, text)

                    if key not in spans:
                        spans[key] = {
                            "contract_title": title,
                            "start": start,
                            "end": end,
                            "text": text,
                            "labels": [],
                        }

                    # Merge multiple category labels for
                    # an identical span.
                    if category not in spans[key]["labels"]:
                        spans[key]["labels"].append(category)

                    stats["valid_answer_annotations"] += 1

    records = list(spans.values())

    # Sort labels and records for reproducibility.
    for record in records:
        record["labels"].sort()

        for category in record["labels"]:
            category_counts[category] += 1

    records.sort(
        key=lambda record: (
            record["contract_title"],
            record["start"],
            record["end"],
            record["text"],
        )
    )

    stats["contracts"] = len(contract_titles)
    stats["unique_annotated_spans"] = len(records)

    stats["spans_with_multiple_labels"] = sum(
        len(record["labels"]) > 1
        for record in records
    )

    stats["spans_with_one_label"] = sum(
        len(record["labels"]) == 1
        for record in records
    )

    stats["unique_categories"] = len(category_counts)

    return {
        "split": split_name,
        "records": records,
        "stats": dict(stats),
        "category_counts": dict(category_counts),
        "contract_titles": sorted(contract_titles),
    }


# ---------------------------------------------------------
# SAVE JSONL
# ---------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    """
    Save one JSON object per line.

    Each record contains:
        contract_title
        start
        end
        text
        labels
    """

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONTRACTSENTINEL - CUAD DATA PREPARATION")
    print("=" * 70)

    train = prepare_spans(
        TRAIN_FILE,
        "train"
    )

    test = prepare_spans(
        TEST_FILE,
        "test"
    )

    # Verify that the official train and test contracts
    # remain separate.
    train_titles = set(train["contract_titles"])
    test_titles = set(test["contract_titles"])

    overlap = train_titles.intersection(test_titles)

    if overlap:
        raise ValueError(
            "Train/test contract overlap detected: "
            + ", ".join(sorted(overlap)[:10])
        )

    # Fail early if annotations were unexpectedly corrupted.
    for result in (train, test):
        stats = result["stats"]

        if stats.get("offset_mismatches", 0) > 0:
            raise ValueError(
                f"{result['split']} contains invalid answer offsets."
            )

        if stats.get("invalid_start_offsets", 0) > 0:
            raise ValueError(
                f"{result['split']} contains invalid start offsets."
            )

    # Save prepared datasets.
    train_path = OUTPUT_DIR / "train_spans.jsonl"
    test_path = OUTPUT_DIR / "test_spans.jsonl"

    save_jsonl(train["records"], train_path)
    save_jsonl(test["records"], test_path)

    summary = {
        "dataset": "CUAD",
        "task": "annotated_span_multilabel_classification",
        "training": {
            "stats": train["stats"],
            "category_counts": train["category_counts"],
        },
        "testing": {
            "stats": test["stats"],
            "category_counts": test["category_counts"],
        },
        "train_test_contract_overlap": len(overlap),
        "limitations": [
            (
                "This dataset contains annotated answer spans, "
                "not all clauses in each contract."
            ),
            (
                "Categories absent from a span's labels are "
                "not automatically verified negative examples."
            ),
            (
                "This dataset does not provide legal risk labels."
            ),
            (
                "Evaluation on these spans does not measure "
                "the ability to locate clauses in full documents."
            ),
        ],
    }

    summary_path = OUTPUT_DIR / "dataset_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    # Print results.
    for result in (train, test):
        print()
        print("-" * 70)
        print(f"SPLIT: {result['split'].upper()}")
        print("-" * 70)

        for name, value in result["stats"].items():
            print(f"{name}: {value}")

        print("\nCATEGORY COUNTS:")

        for category, count in sorted(
            result["category_counts"].items(),
            key=lambda item: (-item[1], item[0])
        ):
            print(f"  {category}: {count}")

    print()
    print("=" * 70)
    print("TRAIN/TEST CONTRACT OVERLAP:", len(overlap))
    print("=" * 70)

    print("\nFiles created:")
    print("  ", train_path)
    print("  ", test_path)
    print("  ", summary_path)

    print("\nPreparation completed successfully.")
    print(
        "No application code, original dataset, "
        "or trained model was modified."
    )


if __name__ == "__main__":
    main()