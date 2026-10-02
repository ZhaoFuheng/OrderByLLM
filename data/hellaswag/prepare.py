"""Prepare HellaSwag in BEIR format (queries.jsonl, corpus.jsonl, test.tsv).

Sampling: 200 queries from the validation split with stdlib `random.seed(42)`.
(The original HellaSwag test split has hidden labels, so the validation split
is the de facto evaluation set used in the literature.)
Corpus: the 4 endings of each sampled claim -> 800 documents.
Qrels: one relevant doc per query (the gold ending), score=1.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).parent
N_QUERIES = 200
SEED = 42

ds = load_dataset("Rowan/hellaswag", split="validation")
print(f"Loaded validation split: {len(ds)} examples")

rng = random.Random(SEED)
indices = sorted(rng.sample(range(len(ds)), N_QUERIES))

queries_path = OUT_DIR / "queries.jsonl"
corpus_path = OUT_DIR / "corpus.jsonl"
qrels_path = OUT_DIR / "test.tsv"

with queries_path.open("w") as qf, corpus_path.open("w") as cf, qrels_path.open("w") as tf:
    tf.write("query-id\tcorpus-id\tscore\n")
    for i in indices:
        ex = ds[i]
        qid = f"hs-{i}"
        qf.write(json.dumps({
            "_id": qid,
            "text": ex["ctx"],
            "metadata": {"activity_label": ex["activity_label"], "source_id": ex["source_id"]},
        }) + "\n")

        gold = int(ex["label"])
        for j, ending in enumerate(ex["endings"]):
            doc_id = f"hs-{i}-{j}"
            cf.write(json.dumps({
                "_id": doc_id,
                "title": ex["activity_label"],
                "text": ending,
                "metadata": {"claim_index": i, "ending_index": j},
            }) + "\n")
            if j == gold:
                tf.write(f"{qid}\t{doc_id}\t1\n")

print(f"Wrote {queries_path.name}, {corpus_path.name}, {qrels_path.name}")
