# DIME

This repository includes the code to replicate the results of the paper ["Dimension Importance Estimation for Dense Information Retrieval"](https://dl.acm.org/doi/pdf/10.1145/3626772.3657691) by Guglielmo Faggioli,  Nicola Ferro, Raffaele Perego, Nicola Tonellotto, presented at SIGIR 2025.


## Prerequisites

Besides the requirements in the `requirements.txt` file, also faiss should be installed. Follow the instructions available [here](https://github.com/facebookresearch/faiss?tab=readme-ov-file) to install it.

To work, the code requires the encoding of the corpora to be already available in a memmap. A memmap is a numpy data structure used to store and read large numpy matrices efficiently. More details are available [here](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).

 To compute the encodings, it is possible to use the method `encode_documents` of the instances of the classes available in the `ir_models.dense` module. 
These memmaps must be saved in:

    data/memmap/<name_of_the_corpus>/<name_of_the_encoder>

Within this directory, there should be two files `<name_of_the_encoder>.dat` that contains the mammap and `<name_of_the_encoder>_map.csv` that should be a csv with two columns `doc_id` and `offset`. The first column is the document id in the corpus, while the second is the offset (i.e., the number of the row) at which the document is encoded in the memmap.

to replicate exactly the results of the paper, `<name_of_the_corpus>`should be one between `msmarco-passages`and `tipster`. While `<name_of_the_encoder>` is one among `ance, contriever, tasb`.
Currently, the repository contains empty files as placeholders for the contriever encodings.

The repository also supports the `qwen3embedding06b` encoder, backed by `Qwen/Qwen3-Embedding-0.6B`. This model uses 1024-dimensional normalized embeddings. Queries are encoded with Qwen's built-in `query` prompt, while documents are encoded without a prompt.

The `bidirlm1bembedding` encoder is backed by `BidirLM/BidirLM-1B-Embedding`. This model uses 1152-dimensional normalized embeddings and requires `trust_remote_code=True` through Sentence Transformers because it uses BidirLM's custom architecture.

The `bgem3` encoder is backed by `BAAI/bge-m3`. This model uses 1024-dimensional normalized dense embeddings.

The `embeddinggemma300m` encoder is backed by `google/embeddinggemma-300m`. This model uses 768-dimensional normalized embeddings, applies Sentence Transformers' query/document retrieval prompts, and requires accepting Google's Gemma usage license on Hugging Face before downloading the weights.

Document memmaps can be generated with:

    python code/encode_memmap.py --corpus msmarco-passages --encoder qwen3embedding06b --batch-size 64
    python code/encode_memmap.py --corpus msmarco-passages --encoder bidirlm1bembedding --batch-size 64
    python code/encode_memmap.py --corpus msmarco-passages --encoder bgem3 --batch-size 64
    python code/encode_memmap.py --corpus msmarco-passages --encoder embeddinggemma300m --batch-size 64

Use `--corpus tipster` to encode the Robust04/Tipster corpus instead. Existing memmap files are not overwritten unless `--overwrite` is passed.

## Ulysses-RFCorpus

The pipeline can also run on [Ulysses-RFCorpus](https://github.com/ulysses-camara/Ulysses-RFCorpus), a relevance feedback dataset for Brazilian legal information retrieval. Download the dataset and place the two CSV files as:

    data/ulysses-rfcorpus/raw/bills_dataset.csv
    data/ulysses-rfcorpus/raw/relevance_feedback_dataset.csv

For this collection, document ids are read from `bills_dataset.csv` column `code`, document text is read from column `text`, and queries are read from `relevance_feedback_dataset.csv` columns `id` and `query`. Relevance feedback is flattened into qrels with `i = 0`, `pr = 1`, and `r = 2`; `extra_results` are included as relevant documents when they match known bill ids.

Generate document memmaps for the four supported recent encoders with:

    python code/encode_memmap.py --corpus ulysses-rfcorpus --encoder qwen3embedding06b --batch-size 64
    python code/encode_memmap.py --corpus ulysses-rfcorpus --encoder bidirlm1bembedding --batch-size 64
    python code/encode_memmap.py --corpus ulysses-rfcorpus --encoder bgem3 --batch-size 64
    python code/encode_memmap.py --corpus ulysses-rfcorpus --encoder embeddinggemma300m --batch-size 64

To run all four encoders with `oracle`, `rel`, and `prf` DIME modes and collect metric CSVs:

    python code/run_ulysses_experiments.py --basepath . --output-dir results/ulysses-rfcorpus

This writes one per-query and one mean metric CSV per encoder/DIME pair, plus:

    results/ulysses-rfcorpus/all_per_query_metrics.csv
    results/ulysses-rfcorpus/all_mean_metrics.csv

For a quick smoke test, pass `--limit` to encode only the first documents and restrict the runner to one encoder and DIME mode:

    python code/run_ulysses_experiments.py --limit 100 --encoders bgem3 --dimes prf --output-dir results/ulysses-rfcorpus-smoke

## Running the code
If you satisfy the prerequisites (i.e., you have a memmap data structure that stores the encoding of all the documents in the corpus in the directory, as described above), the code can run from the main directory with the following line:

    python code/main.py -c <name_of_the_collection> -e <name_of_the_encoder> -d <name_of_the_dime>

Name of the collection should be one among `trec-dl-2019`, `trec-dl-2020`, `trec-robust-2004`, `ulysses-rfcorpus`.
Concerning the name of the dime, it should be one among `oracle`, `rel`, `llm`, `prf`.

For example, after generating the Qwen memmap for MS MARCO passages:

    python code/main.py -c trec-dl-2019 -e qwen3embedding06b -d prf

Or, after generating the BidirLM memmap:

    python code/main.py -c trec-dl-2019 -e bidirlm1bembedding -d prf

Additional supported encoders can be selected in the same way:

    python code/main.py -c trec-dl-2019 -e bgem3 -d prf
    python code/main.py -c trec-dl-2019 -e embeddinggemma300m -d prf


### Citing this work
To cite this work, use the following bib entry:

    @inproceedings{DBLP:conf/sigir/Faggioli00T24,
        author       = {Guglielmo Faggioli and
                  Nicola Ferro and
                  Raffaele Perego and
                  Nicola Tonellotto},
        editor       = {Grace Hui Yang and
                  Hongning Wang and
                  Sam Han and
                  Claudia Hauff and
                  Guido Zuccon and
                  Yi Zhang},
        title        = {Dimension Importance Estimation for Dense Information Retrieval},
        booktitle    = {Proceedings of the 47th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2024, Washington
                  DC, USA, July 14-18, 2024},
        pages        = {1318--1328},
        publisher    = {{ACM}},
        year         = {2024},
        url          = {https://doi.org/10.1145/3626772.3657691},
        doi          = {10.1145/3626772.3657691},
    }
  

