import argparse
import ast
import csv
import json
import math
import random
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data") / "ulysses-rfcorpus" / "raw"
DEFAULT_OUTPUT_DIR = Path("data") / "ulysses-rfcorpus-subset" / "raw"


def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_id(value):
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return str(int(value))

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def parse_serialized(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

    return None


def feedback_items(value):
    parsed = parse_serialized(value)
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if "id" in parsed and "class" in parsed:
            if isinstance(parsed["id"], (list, tuple)):
                return [
                    {"id": doc_id, "class": rel_class}
                    for doc_id, rel_class in zip(parsed["id"], parsed["class"])
                ]
            return [parsed]

        if all(isinstance(v, (list, tuple)) for v in parsed.values()) and {"id", "class"} <= set(parsed):
            return [
                {"id": doc_id, "class": rel_class}
                for doc_id, rel_class in zip(parsed["id"], parsed["class"])
            ]

    if isinstance(parsed, (list, tuple)):
        return [item for item in parsed if isinstance(item, dict)]

    return []


def extra_result_ids(value):
    parsed = parse_serialized(value)
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if "id" in parsed:
            return extra_result_ids(parsed["id"])
        return []

    if isinstance(parsed, (list, tuple, set)):
        ids = []
        for item in parsed:
            if isinstance(item, dict):
                ids.extend(extra_result_ids(item))
            else:
                doc_id = normalize_id(item)
                if doc_id is not None:
                    ids.append(doc_id)
        return ids

    doc_id = normalize_id(parsed)
    return [doc_id] if doc_id is not None else []


def read_bills(path):
    rows_by_code = {}
    aliases = {}
    lengths = {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = normalize_id(row.get("code"))
            text = row.get("text") or ""
            if code is None or not text or code in rows_by_code:
                continue

            rows_by_code[code] = row
            lengths[code] = len(text)

            name = normalize_id(row.get("name"))
            if name is not None:
                aliases[name] = code

    return rows_by_code, aliases, lengths, reader.fieldnames


def read_feedback(path):
    seen_query_ids = set()
    rows = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            query_id = normalize_id(row.get("id"))
            if query_id is None or query_id in seen_query_ids:
                continue

            seen_query_ids.add(query_id)
            row["_query_id"] = query_id
            rows.append(row)

    return rows, fieldnames


def resolve_doc_id(value, aliases):
    doc_id = normalize_id(value)
    if doc_id is None:
        return None, None
    return aliases.get(doc_id, doc_id), doc_id


def relevant_bill_ids(feedback_rows, aliases, valid_bill_ids, relevant_classes):
    relevant_ids = set()
    unknown_ids = set()

    for row in feedback_rows:
        for item in feedback_items(row.get("user_feedback")):
            relevance_class = normalize_id(item.get("class"))
            if relevance_class is None or relevance_class.lower() not in relevant_classes:
                continue

            doc_id, raw_doc_id = resolve_doc_id(item.get("id"), aliases)
            if doc_id is None:
                continue
            if doc_id in valid_bill_ids:
                relevant_ids.add(doc_id)
            else:
                unknown_ids.add(raw_doc_id)

        for raw_extra_id in extra_result_ids(row.get("extra_results")):
            doc_id, raw_doc_id = resolve_doc_id(raw_extra_id, aliases)
            if doc_id is None:
                continue
            if doc_id in valid_bill_ids:
                relevant_ids.add(doc_id)
            else:
                unknown_ids.add(raw_doc_id)

    return relevant_ids, unknown_ids


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_subset(args):
    set_csv_field_limit()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    bills_path = input_dir / "bills_dataset.csv"
    feedback_path = input_dir / "relevance_feedback_dataset.csv"

    rows_by_code, aliases, lengths, bill_fieldnames = read_bills(bills_path)
    feedback_rows, feedback_fieldnames = read_feedback(feedback_path)

    if args.num_queries > len(feedback_rows):
        raise ValueError(f"requested {args.num_queries} queries, but only {len(feedback_rows)} are available")

    rng = random.Random(args.seed)
    sampled_feedback = rng.sample(feedback_rows, args.num_queries)
    relevant_classes = {rel.lower() for rel in args.relevant_classes}
    relevant_ids, unknown_relevant_ids = relevant_bill_ids(
        sampled_feedback,
        aliases,
        set(rows_by_code),
        relevant_classes,
    )

    random_pool = [
        doc_id
        for doc_id, length in lengths.items()
        if doc_id not in relevant_ids and length <= args.max_random_bill_chars
    ]
    rng.shuffle(random_pool)

    random_needed = max(0, args.target_bills - len(relevant_ids))
    random_ids = set(random_pool[:random_needed])
    subset_ids = relevant_ids | random_ids

    output_dir.mkdir(parents=True, exist_ok=True)

    bill_rows = [rows_by_code[doc_id] for doc_id in rows_by_code if doc_id in subset_ids]
    feedback_output_rows = [
        {key: value for key, value in row.items() if key != "_query_id"} for row in sampled_feedback
    ]

    write_csv(output_dir / "bills_dataset.csv", bill_fieldnames, bill_rows)
    write_csv(output_dir / "relevance_feedback_dataset.csv", feedback_fieldnames, feedback_output_rows)

    metadata = {
        "seed": args.seed,
        "num_queries": args.num_queries,
        "target_bills": args.target_bills,
        "max_random_bill_chars": args.max_random_bill_chars,
        "relevant_classes": sorted(relevant_classes),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "available_queries": len(feedback_rows),
        "available_bills": len(rows_by_code),
        "sampled_query_ids": [row["_query_id"] for row in sampled_feedback],
        "known_relevant_bills": len(relevant_ids),
        "random_bills_added": len(random_ids),
        "subset_bills": len(subset_ids),
        "unknown_relevant_ids": len(unknown_relevant_ids),
        "unknown_relevant_examples": sorted(unknown_relevant_ids)[:20],
        "oversized_relevant_bills_kept": sum(
            1 for doc_id in relevant_ids if lengths[doc_id] > args.max_random_bill_chars
        ),
    }
    with (output_dir / "subset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a smaller Ulysses-RFCorpus subset for faster model tests."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--target-bills", type=int, default=10000)
    parser.add_argument("--max-random-bill-chars", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relevant-classes", nargs="+", default=["pr", "r"])
    return parser.parse_args()


def main():
    metadata = create_subset(parse_args())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
