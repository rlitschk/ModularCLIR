"""
This script downloads MS-MARCO passage retrieval data and prepares them for
dataset.load_from_pretrained('json',...) from huggingface. It writes the dev set
and train set as jsonl files.

Dataloaders in this code are wrappers for the code made available by Reimers et al.
in their sentence-transformers library:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py
"""
import gzip
import json
import logging
import os
import tarfile
import tqdm
import argparse

from collections import defaultdict
from typing import Tuple, List

from sentence_transformers import util, InputExample


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="Target folder for storing prepared data files.")
args = parser.parse_args()


def get_corpus(cache_dir: str) -> dict:
  #### Read the corpus files, that contain all the passages. Store them in the corpus dict
  print("loading corpus")
  corpus = {}
  collection_filepath = os.path.join(cache_dir, 'collection.tsv')
  if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(cache_dir, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
      logging.info("Download collection.tar.gz")
      util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
      tar.extractall(path=cache_dir)
  with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
      pid, passage = line.strip().split("\t")
      corpus[pid] = passage
  return corpus


def get_queries(cache_dir: str) -> dict:
  ### Read the train queries, store in queries dict (extended by us, we read all queries, needed for cross-check dev-set)
  queries = {}
  for split in ["train", "dev", "eval"]:
    fName = f'queries.{split}.tsv'
    print(f"loading {fName}")
    queries_filepath = os.path.join(cache_dir, fName)
    if not os.path.exists(queries_filepath):
      tar_filepath = os.path.join(cache_dir, 'queries.tar.gz')
      if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

      with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=cache_dir)
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
      for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query
  assert len(queries) == 1010916
  return queries


def get_dev_samples(
    queries: dict,
    corpus: dict,
    sbert_splits: bool,
    cache_dir: str,
) -> Tuple[dict, list]:
  """
  sbert split:
  - We use 200 random queries from the train set for evaluation during training
  - Each query has at least one relevant and up to 200 irrelevant (negative) passages

  - msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
  shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
  - We extracted in the train-eval split 500 random queries that can be used for evaluation during training

  Otherwise:
  - loads full ms-marco dev dataset

  :param queries: query-id to query mapping
  :param corpus: passage-id to passage mapping
  :param sbert_splits: whether to use the data prepared by Reimers et al. or original (but much larger) msmarco data
  :return: mapping dev_samples from each query to all positive/negative passages and jsonl formatted lines
  """

  if not sbert_splits:
    # Load full development set from ms-marco: loads top100dev and qrels.dev.small.tsv (containing only queries from top1000 dev)
    download_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz'
    filename = 'top1000.dev.tar.gz'

    # Load labels file: {qrels.dev.small.tsv} inside {collectionandqueries.tar.gz} (7437 lines)
    qrels_filepath = os.path.join(cache_dir, "qrels.dev.small.tsv")
    if not os.path.exists(qrels_filepath.replace(".tar.gz","/qrels.dev.small.tsv")):
      container_filepath = os.path.join(cache_dir, "collectionandqueries.tar.gz")
      if not os.path.exists(container_filepath):
        logging.info("Download " + os.path.basename(container_filepath))
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz', container_filepath)
      with tarfile.open(container_filepath, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    # Process labels file
    qid2relevant_pids = defaultdict(list)
    with open(qrels_filepath, encoding="UTF-8") as fIn:
      for line in tqdm.tqdm(fIn, total=7437, desc="Loading relevance labels"):
        # We ignore columns 1 and 3 as they are there for TREC formating.
        qid, _, pid, _ = line.strip().split("\t")
        assert qid in queries # query id should be included in set of all queries
        assert pid in corpus
        qid2relevant_pids[qid].append(pid)

    # load {top1000.dev} (6668967 lines)
    top1000dev = os.path.join(cache_dir, filename.replace(".tar.gz", ""))
    if not os.path.exists(top1000dev):
      train_eval_filepath = os.path.join(cache_dir, filename)
      if not os.path.exists(train_eval_filepath):
        logging.info("Download " + os.path.basename(train_eval_filepath))
        util.http_get(download_url, train_eval_filepath)
      with tarfile.open(train_eval_filepath, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    dev_lines = []
    dev_samples = {}
    with open(top1000dev, encoding="UTF-8") as fIn:
      for line in tqdm.tqdm(fIn, total=6668967, desc="Creating .jsonl dev dataset"):
        qid, pid, query, passage = line.strip().split("\t")
        label = 1 if pid in qid2relevant_pids[qid] else 0
        assert qid in queries
        assert pid in corpus
        sample = {"qid": qid, "query": query, "pid": pid, "passage": passage, "label": label}
        dev_lines.append(sample)
    return dev_samples, dev_lines

  else:
    # Take distinct subset of training triples for dev set
    num_dev_queries = 200
    num_max_dev_negatives = 200
    download_url = 'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz'
    filename = 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz'

  # train_eval_filepath = os.path.join(cache_dir, filename)
  train_eval_filepath = os.path.join(cache_dir, filename)
  if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get(download_url, train_eval_filepath)

  dev_samples = {}
  dev_lines = []
  print(f"loading {train_eval_filepath}")
  with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, total=20000000):
      qid, pos_id, neg_id = line.strip().split()
      query = queries[qid]

      if qid not in dev_samples and len(dev_samples) < num_dev_queries:
        dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

      if qid in dev_samples:
        positive = corpus[pos_id]
        dev_samples[qid]['positive'].add(positive)
        dev_lines.append({"qid": qid, "query": query, "pid": pos_id, "passage": positive, "label": 1})

        if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
          negative = corpus[neg_id]
          dev_samples[qid]['negative'].add(negative)
          dev_lines.append({"qid": qid, "query": query, "pid": neg_id, "passage": negative, "label": 0})
  return dev_samples, dev_lines


def get_train_samples(
    corpus: dict,
    queries: dict,
    dev_samples: dict,
    sbert_splits: bool,
    cache_dir: str,
) -> Tuple[list, list]:
  """
  We train the network with as a binary label task
  Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
  We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
  in our training setup. For the negative samples, we use the triplets provided by MS Marco that
  specify (query, positive sample, negative sample).
  :param corpus: pid2passage dictionary
  :param queries: qid2query dictionary
  :param dev_samples: used to ensure no query overlap between dev and train
  :param sbert_splits: whether to use the data prepared by Reimers et al. or original (but much larger) msmarco data
  :param cache_dir: download directory and save directory for cache files
  :return:
  """
  train_samples = []
  train_lines = []

  if sbert_splits:
    # number of negative passages for each positive passage
    pos_neg_ratio = 4
    # Maximal number of training samples we want to use
    max_train_samples = 2e7
    train_file = 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz'
    base_url = 'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz'
    total = None
  else:
    pos_neg_ratio = 1
    max_train_samples = -1 # all
    train_file = 'qidpidtriples.train.full.2.tsv.gz'
    base_url = 'https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz'
    total = 397768673

  train_filepath = os.path.join(cache_dir, train_file)
  if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get(base_url, train_filepath)

  cnt = 0
  with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True, total=total):
      qid, pos_id, neg_id = line.strip().split()
      assert pos_id in corpus
      assert neg_id in corpus
      assert qid in corpus

      if qid in dev_samples:
        continue

      query = queries[qid]

      if sbert_splits:
        if (cnt % (pos_neg_ratio + 1)) == 0:
          passage = corpus[pos_id]
          label = 1
          pid = pos_id
        else:
          passage = corpus[neg_id]
          label = 0
          pid = neg_id

        train_samples.append(InputExample(texts=[query, passage], label=label))
        train_lines.append({"qid": qid, "query": query, "pid": pid, "passage": passage, "label": label})
        cnt += 1
        if cnt >= max_train_samples:
          break
      else:
        train_samples.append(InputExample(texts=[query, corpus[pos_id]], label=1))
        train_samples.append(InputExample(texts=[query, corpus[neg_id]], label=0))
        train_lines.append({"qid": qid, "query": query, "pid": pos_id, "passage": corpus[pos_id], "label": 1})
        train_lines.append({"qid": qid, "query": query, "pid": neg_id, "passage": corpus[neg_id], "label": 0})

  return train_samples, train_lines


def maybe_save_jsonl(lines: List[dict], filepath: str, overwrite:bool=False):
  if not os.path.exists(filepath) or overwrite:
    path, file = os.path.split(filepath)
    os.makedirs(path, exist_ok=True)
    print(f"Saving {filepath}")
    with open(filepath, "w") as f:
      for line in tqdm.tqdm(lines):
        f.write(json.dumps(line) + "\n")


def main():
  sbert_spits = True
  cache_dir = "data/ms-marco/"
  target_folder = args.output_dir
  os.makedirs(target_folder, exist_ok=True)
  
  # load ms-marco
  corpus = get_corpus(cache_dir)
  queries = get_queries(cache_dir)
  
  # load dev split from reimers
  dev_samples, dev_lines = get_dev_samples(queries, corpus, sbert_spits, cache_dir)
  maybe_save_jsonl(dev_lines, target_folder + "dev%s.jsonl" % ("_sbert" if sbert_spits else ""))
  
  # load train split from reimers
  train_samples, train_lines = get_train_samples(corpus, queries, dev_samples, sbert_spits, cache_dir)
  maybe_save_jsonl(train_lines, target_folder + "train%s.jsonl" % ("_sbert" if sbert_spits else ""))
  

if __name__ == '__main__':
  main()
