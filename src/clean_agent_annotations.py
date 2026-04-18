"""
Clean agent-labeled JSONL for bert_train.py.

Only two cleaning operations are applied:
1) Remove failed IDs listed in error log.
2) De-duplicate by ID (keep the last occurrence).

Then export CSV with columns compatible with src/bert_train.py:
- pid
- title
- description
- parameters
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean agent annotation outputs")
    parser.add_argument(
        "--input-jsonl",
        default="output/agent_bio_annotations.jsonl",
        help="Input annotation JSONL file",
    )
    parser.add_argument(
        "--error-log",
        default="output/agent_bio_errors.log",
        help="Error log file (failed IDs)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/agent_labeled_for_bert_train.csv",
        help="Output CSV for bert_train.py",
    )
    parser.add_argument(
        "--output-jsonl",
        default="output/agent_bio_annotations.cleaned.jsonl",
        help="Output cleaned JSONL",
    )
    return parser.parse_args()


def load_failed_ids(error_log_path: Path) -> Set[str]:
    if not error_log_path.exists():
        return set()
    failed = set()
    with error_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # expected format: <id>\t<ErrorType>\t<message>
            failed.add(line.split("\t", 1)[0].strip())
    return failed


def normalize_entity_list(values: List[str]) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned = []
    seen = set()
    for v in values:
        s = str(v).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            cleaned.append(s)
    return cleaned


def to_parameters_string(entities: Dict[str, List[str]]) -> str:
    # Keep material/brand as single value for stable comma-split parsing in bert_train.py.
    materials = normalize_entity_list(entities.get("material", []))
    colors = normalize_entity_list(entities.get("color", []))
    brands = normalize_entity_list(entities.get("brand", []))

    material = materials[0] if materials else ""
    color = ", ".join(colors) if colors else ""
    brand = brands[0] if brands else ""

    # bert_train.py only needs fields index 2/3/4 after split by ", "
    return f"Unknown, Unknown, {material}, {color}, {brand}"


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    error_log = Path(args.error_log)
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    failed_ids = load_failed_ids(error_log)

    # Deduplicate by ID, keep the last record.
    dedup_records: Dict[str, Dict] = {}
    total_lines = 0
    skipped_failed = 0
    skipped_no_id = 0
    parse_errors = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            row_id = str(record.get("id", "")).strip()
            if not row_id:
                skipped_no_id += 1
                continue
            if row_id in failed_ids:
                skipped_failed += 1
                continue
            dedup_records[row_id] = record

    cleaned_records = list(dedup_records.values())

    # Write cleaned JSONL.
    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in cleaned_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write bert_train compatible CSV.
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pid", "title", "description", "parameters"]
        )
        writer.writeheader()
        for rec in cleaned_records:
            canonical = rec.get("pred_entities_canonical", {})
            writer.writerow(
                {
                    "pid": rec.get("id", ""),
                    "title": rec.get("title", ""),
                    "description": rec.get("description", ""),
                    "parameters": to_parameters_string(canonical),
                }
            )

    print(
        json.dumps(
            {
                "input_lines": total_lines,
                "failed_ids_in_log": len(failed_ids),
                "skipped_failed": skipped_failed,
                "skipped_no_id": skipped_no_id,
                "json_parse_errors": parse_errors,
                "output_records": len(cleaned_records),
                "cleaned_jsonl": str(output_jsonl),
                "bert_train_csv": str(output_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
