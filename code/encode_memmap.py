import argparse
import os
import sys
from pathlib import Path

import ir_datasets
import numpy as np

sys.path += [".", "DIME_simple/code", "DIME_simple/code/ir_models"]

from datasets.ulysses_rfcorpus import ULYSSES_CORPUS_NAME, UlyssesCorpus
from ir_models.dense import get_dense_model


CORPORA = {
    "msmarco-passages": "msmarco-passage",
    "tipster": "disks45/nocr",
    ULYSSES_CORPUS_NAME: None,
}


def get_document_text(doc):
    if hasattr(doc, "text"):
        return doc.text

    parts = []
    for field in ["title", "body"]:
        if hasattr(doc, field):
            value = getattr(doc, field)
            if value:
                parts.append(value)

    if parts:
        return " ".join(parts)

    raise ValueError(f"could not find text fields for document {doc.doc_id}")


def get_docs_count(dataset):
    if hasattr(dataset, "docs_count"):
        count = dataset.docs_count()
        if count is not None:
            return count

    return sum(1 for _ in dataset.docs_iter())


def iter_batches(dataset, batch_size, limit=None):
    batch_ids = []
    batch_texts = []

    for offset, doc in enumerate(dataset.docs_iter()):
        if limit is not None and offset >= limit:
            break

        batch_ids.append(doc.doc_id)
        batch_texts.append(get_document_text(doc))

        if len(batch_ids) == batch_size:
            yield batch_ids, batch_texts
            batch_ids = []
            batch_texts = []

    if batch_ids:
        yield batch_ids, batch_texts


def encode_memmap(args):
    if args.corpus == ULYSSES_CORPUS_NAME:
        dataset = UlyssesCorpus(basepath=args.basepath)
    else:
        dataset = ir_datasets.load(CORPORA[args.corpus])

    total_docs = get_docs_count(dataset)

    if args.limit is not None:
        total_docs = min(total_docs, args.limit)

    encoder = get_dense_model(args.encoder)
    embedding_dim = encoder.get_embedding_dim()

    output_dir = Path(args.basepath) / "data" / "memmap" / args.corpus / args.encoder
    output_dir.mkdir(parents=True, exist_ok=True)

    memmap_path = output_dir / f"{args.encoder}.dat"
    mapping_path = output_dir / f"{args.encoder}_map.csv"

    if not args.overwrite and (memmap_path.exists() or mapping_path.exists()):
        raise FileExistsError(f"{output_dir} already contains memmap files; pass --overwrite to replace them")

    data = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(total_docs, embedding_dim))

    offset = 0
    with open(mapping_path, "w") as fp:
        fp.write("doc_id,offset\n")

        for batch_ids, batch_texts in iter_batches(dataset, args.batch_size, args.limit):
            embeddings = encoder.encode_documents(batch_texts).astype(np.float32)
            next_offset = offset + len(batch_ids)
            data[offset:next_offset] = embeddings

            for doc_id in batch_ids:
                fp.write(f"{doc_id},{offset}\n")
                offset += 1

            data.flush()

    if offset != total_docs:
        raise ValueError(f"encoded {offset} documents, but expected {total_docs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--basepath", default=".")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    encode_memmap(args)
