import ast
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ULYSSES_CORPUS_NAME = "ulysses-rfcorpus"
ULYSSES_RAW_DIR = Path("data") / ULYSSES_CORPUS_NAME / "raw"
BILLS_FILENAME = "bills_dataset.csv"
RELEVANCE_FEEDBACK_FILENAME = "relevance_feedback_dataset.csv"
RELEVANCE_MAP = {"i": 0, "pr": 1, "r": 2}


def _raw_dir(basepath):
    return Path(basepath) / ULYSSES_RAW_DIR


def _bills_path(basepath):
    return _raw_dir(basepath) / BILLS_FILENAME


def _feedback_path(basepath):
    return _raw_dir(basepath) / RELEVANCE_FEEDBACK_FILENAME


def _require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Download Ulysses-RFCorpus and place {BILLS_FILENAME} "
            f"and {RELEVANCE_FEEDBACK_FILENAME} under {path.parent}"
        )


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


def _read_bills(basepath):
    path = _bills_path(basepath)
    _require_file(path)

    bills = pd.read_csv(path)
    required = {"code", "text"}
    missing = required - set(bills.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    columns = ["code", "text"]
    if "name" in bills.columns:
        columns.append("name")

    docs = bills[columns].rename(columns={"code": "doc_id"}).copy()
    docs["doc_id"] = docs.doc_id.map(normalize_id)
    docs["text"] = docs.text.fillna("").astype(str)
    if "name" in docs.columns:
        docs["name"] = docs.name.map(normalize_id)
    docs = docs.dropna(subset=["doc_id"]).drop_duplicates(subset=["doc_id"], keep="first")
    docs = docs.loc[docs.text.str.len() > 0].reset_index(drop=True)
    return docs


def iter_ulysses_documents(basepath):
    docs = _read_bills(basepath)
    for row in docs.itertuples(index=False):
        yield SimpleNamespace(doc_id=row.doc_id, text=row.text)


@dataclass
class UlyssesCorpus:
    basepath: str = "."

    def docs_iter(self):
        return iter_ulysses_documents(self.basepath)

    def docs_count(self):
        return len(_read_bills(self.basepath))


def _parse_serialized(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (dict, list, tuple)):
        return value

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

    return None


def _feedback_items(value):
    parsed = _parse_serialized(value)
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


def _extra_result_ids(value):
    parsed = _parse_serialized(value)
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if "id" in parsed:
            return _extra_result_ids(parsed["id"])
        return []

    if isinstance(parsed, (list, tuple, set)):
        ids = []
        for item in parsed:
            if isinstance(item, dict):
                ids.extend(_extra_result_ids(item))
            else:
                doc_id = normalize_id(item)
                if doc_id is not None:
                    ids.append(doc_id)
        return ids

    doc_id = normalize_id(parsed)
    return [doc_id] if doc_id is not None else []


def _resolve_doc_id(value, aliases):
    doc_id = normalize_id(value)
    if doc_id is None:
        return None, None
    return aliases.get(doc_id, doc_id), doc_id


def _flatten_qrels(feedback, valid_doc_ids, doc_id_aliases=None):
    doc_id_aliases = doc_id_aliases or {}
    rows = []
    unknown_doc_ids = set()

    for record in feedback.itertuples(index=False):
        query_id = normalize_id(getattr(record, "id"))
        if query_id is None:
            continue

        for item in _feedback_items(getattr(record, "user_feedback", None)):
            doc_id, raw_doc_id = _resolve_doc_id(item.get("id"), doc_id_aliases)
            rel_class = normalize_id(item.get("class"))
            if doc_id is None or rel_class is None:
                continue

            relevance = RELEVANCE_MAP.get(rel_class.lower())
            if relevance is None:
                continue

            if doc_id not in valid_doc_ids:
                unknown_doc_ids.add(raw_doc_id)
                continue

            rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": relevance})

        for raw_extra_id in _extra_result_ids(getattr(record, "extra_results", None)):
            doc_id, raw_doc_id = _resolve_doc_id(raw_extra_id, doc_id_aliases)
            if doc_id is None:
                continue
            if doc_id not in valid_doc_ids:
                unknown_doc_ids.add(raw_doc_id)
                continue
            rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": RELEVANCE_MAP["r"]})

    qrels = pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])
    if not qrels.empty:
        qrels = qrels.groupby(["query_id", "doc_id"], as_index=False).relevance.max()

    if unknown_doc_ids:
        warnings.warn(
            f"dropped {len(unknown_doc_ids)} Ulysses qrel document ids that are not present in {BILLS_FILENAME}",
            RuntimeWarning,
        )

    return qrels


def load_ulysses_queries_qrels(basepath="."):
    feedback_path = _feedback_path(basepath)
    _require_file(feedback_path)

    docs = _read_bills(basepath)
    valid_doc_ids = set(docs.doc_id)
    doc_id_aliases = {}
    if "name" in docs.columns:
        doc_id_aliases = {
            name: doc_id
            for name, doc_id in zip(docs.name, docs.doc_id)
            if name is not None
        }

    feedback = pd.read_csv(feedback_path)
    required = {"id", "query", "user_feedback"}
    missing = required - set(feedback.columns)
    if missing:
        raise ValueError(f"{feedback_path} is missing required columns: {sorted(missing)}")

    queries = feedback[["id", "query"]].rename(columns={"id": "query_id", "query": "text"}).copy()
    queries["query_id"] = queries.query_id.map(normalize_id)
    queries["text"] = queries.text.fillna("").astype(str)
    queries = queries.dropna(subset=["query_id"]).drop_duplicates(subset=["query_id"], keep="first")

    qrels = _flatten_qrels(feedback, valid_doc_ids, doc_id_aliases)
    if qrels.empty:
        raise ValueError("no Ulysses qrels could be loaded from relevance feedback")

    queries = queries.loc[queries.query_id.isin(qrels.query_id)].reset_index(drop=True)
    qrels = qrels.loc[qrels.query_id.isin(queries.query_id)].reset_index(drop=True)

    return queries, qrels
