import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import ir_datasets
import argparse
import importlib
import dime.utils
import sys
from pathlib import Path
from multiprocessing.dummy import Pool
from tqdm.auto import tqdm
sys.path += [".", "DIME_simple/code", "DIME_simple/code/ir_models"]

from datasets.ulysses_rfcorpus import ULYSSES_CORPUS_NAME, load_ulysses_queries_qrels
from ir_models.dense import get_dense_model
import local_utils


COLLECTION_TO_CORPUS = {
    "trec-dl-2019": "msmarco-passages",
    "trec-dl-2020": "msmarco-passages",
    "trec-robust-2004": "tipster",
    ULYSSES_CORPUS_NAME: ULYSSES_CORPUS_NAME,
}

MEASURES = ["AP", "R@1000", "MRR", "nDCG@3", "nDCG@10", "nDCG@100", "nDCG@20", "nDCG@50"]


def load_collection(collection, basepath):
    if collection == "trec-dl-2019":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")

    elif collection == "trec-dl-2020":
        dataset = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")

    elif collection == "trec-robust-2004":
        dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
        qrels = pd.DataFrame(dataset.qrels_iter())
        queries = pd.DataFrame(dataset.queries_iter()).query("query_id in @qrels.query_id")[["query_id", "title"]].rename({"title": "text"}, axis=1)

    elif collection == ULYSSES_CORPUS_NAME:
        queries, qrels = load_ulysses_queries_qrels(basepath)

    else:
        available = ", ".join(sorted(COLLECTION_TO_CORPUS))
        raise ValueError(f"collection not recognized: {collection}. Available collections: {available}")

    queries = queries.copy()
    qrels = qrels.copy()
    queries["query_id"] = queries.query_id.astype(str)
    qrels["query_id"] = qrels.query_id.astype(str)
    qrels["doc_id"] = qrels.doc_id.astype(str)
    return queries, qrels


def save_metrics(per_query, mean_metrics, args):
    if args.output_dir is None:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.collection}_{args.encoder}_{args.dime}"
    per_query.to_csv(output_dir / f"{stem}_per_query.csv", index=False)
    mean_metrics.to_csv(output_dir / f"{stem}_mean.csv", index=False)


def run_pipeline(args):
    queries, qrels = load_collection(args.collection, args.basepath)

    encoder = get_dense_model(args.encoder, max_seq_length=args.max_seq_length)
    tqdm.write(f"encoding {len(queries)} queries with {args.encoder}")
    queries["representation"] = list(encoder.encode_queries(queries.text.to_list()))


    #We assume that you have already computed the memmaps containing the representation for all the documents of the corpus
    #please, visit https://numpy.org/doc/stable/reference/generated/numpy.memmap.html to learn more about memmap.
    #to construct the encoding of the documents, you can use the method encode_documents of the istances of classes in the module ir_models.dense
    #since memmaps do not allow to store the id of the document corresponding to a certain row, we assume this mapping to be stored in a csv file

    memmap_path = f"{args.basepath}/data/memmap/{COLLECTION_TO_CORPUS[args.collection]}/{args.encoder}/{args.encoder}.dat"
    memmap_idmp = f"{args.basepath}/data/memmap/{COLLECTION_TO_CORPUS[args.collection]}/{args.encoder}/{args.encoder}_map.csv"

    docs_encoder = local_utils.MemmapEncoding(memmap_path, memmap_idmp, embedding_size=encoder.get_embedding_dim(), index_name="doc_id")
    indexWrapper = local_utils.FaissIndex(data=docs_encoder.get_data(), mapper=docs_encoder.get_ids())


    if args.dime == "oracle":
        dime_params = {"qrels": qrels, "docs_encoder": docs_encoder}
    elif args.dime == "rel":
        dime_params = {"qrels": qrels, "docs_encoder": docs_encoder}
    elif args.dime == "prf":
        tqdm.write("retrieving initial PRF run")
        run = indexWrapper.retrieve(queries)
        dime_params = {"docs_encoder": docs_encoder, "k": 5, "run": run}
    elif args.dime == "llm":
        # we assume you already have access to a csv (as the one available in the data directory) with llms answers
        answers = pd.read_csv(f"{args.basepath}/data/gpt4_answers.csv").query("query_id in @queries.query_id")
        answers["representation"] = list(encoder.encode_queries(answers.response.to_list()))
        dime_params = {"llm_docs": answers}

    else:
        raise ValueError("dime not recognized")


    dim_estimator = getattr(importlib.import_module(f"dime"), args.dime.capitalize())(**dime_params)
    importance = dim_estimator.compute_importance(queries)


    def alpha_retrieve(parallel_args):
        importance, queries, alpha = parallel_args
        masked_qembs, r2q = dime.utils.get_masked_encoding(queries, importance, alpha)
        run = indexWrapper.retrieve(masked_qembs, r2q)
        run["alpha"] = alpha
        return run


    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    with Pool(processes=len(alphas)) as pool:
        run = pd.concat(
            tqdm(
                pool.imap(alpha_retrieve, [[importance, queries, a] for a in alphas]),
                total=len(alphas),
                desc="retrieving alpha runs",
                unit="alpha",
                dynamic_ncols=True,
            )
        )

    tqdm.write("computing metrics")
    perf = run.groupby("alpha").apply(
        lambda x: local_utils.compute_measure(x, qrels, MEASURES)) \
        .reset_index().drop("level_1", axis=1)

    perf["collection"] = args.collection
    perf["encoder"] = args.encoder
    perf["dime"] = args.dime

    mean_metrics = perf.groupby(["collection", "encoder", "dime", "alpha", "measure"]).value.mean().reset_index()
    save_metrics(perf, mean_metrics, args)

    print(mean_metrics.sort_values(["measure", "alpha"], ascending=True)
          .pivot_table(index="measure", columns="alpha", values="value").to_string())

    return perf, mean_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", type=str, choices=sorted(COLLECTION_TO_CORPUS), required=True)
    parser.add_argument("-e", "--encoder", type=str, required=True)
    parser.add_argument("-d", "--dime", type=str, required=True)
    parser.add_argument("--basepath", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-seq-length", type=int)
    args = parser.parse_args()

    run_pipeline(args)