import pickle
import sys
import os
import torch
import time
import logging
import numpy as np
from scipy.stats import ttest_ind
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
from clef_dataloader import load_clef_rerank
from sentence_transformers import CrossEncoder


logging.basicConfig(
  format="%(asctime)s [%(levelname)s]: %(message)s",
  datefmt="%d.%m.%Y %H:%M:%S",
  handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# precision for MAP rounding
map_ndigits = 3

# Number of documents to re-rank
topk = 100 # args.topk
batch_size = 20
assert topk % batch_size == 0


AP_T, MAP_T, PVAL_T = float, float, float
Q_ID, DOC_ID, LAYER_T = Union[str, int], str, int
MAP_PVAL = Union[Tuple[MAP_T, PVAL_T], Tuple[MAP_T, PVAL_T, List[AP_T]]]


def map2str(eval_map: float, pvalue: float):
  """
  Uniform string formatting of mean average precision (MAP) with significance marker (*) if MAP is significant.
  :param eval_map: mean average precision value
  :param pvalue: 
  :return: 
  """
  map_str = str(round(eval_map, map_ndigits))
  n_zeros = (map_ndigits + 2) - len(map_str)
  significance_marker = "*" if 0 < pvalue <= 0.05 else ""
  map_str = map_str + n_zeros * '0' + significance_marker
  return map_str


def add_filehandler(logfile: str):
  """Utility function for logging"""
  pass
  # fh = logging.FileHandler(filename=logfile)
  # fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s"))
  # logger.addHandler(fh)
  # logger.info(f"logging results to file: {logfile}")


def _get_rerank_dir(dlang: str, preranking_model: str, qlang: str, prerank_dir: str):
  """
  Monolingual:
  bm25, fbmt+bm25, marianmt+bm25, unigram, fasttext
  
  Cross-lingual: 
  procb, distil_mbert, ...
  
  Given a model (@model) and language pair of query language (@qlang) and document language (@dlang) return directory
  containing all ranking files. Each ranking file corresponds to a list of document ids for a given query id (=filename). 
  """
  preranking_model = preranking_model.lower()

  # Path to monolingual ranking files (or those that have been translated)
  if preranking_model == "bm25":
    assert qlang == dlang
    rerank_dir = os.path.join(prerank_dir, "mono/bm25/%s-%s/" % (qlang, dlang)) 
  elif preranking_model == "fbmt+bm25":
    rerank_dir = os.path.join(prerank_dir, "xling/fbnm25+bm25/%s-%s/" % (qlang, dlang))
  elif preranking_model == "marianmt+bm25":
    rerank_dir = os.path.join(prerank_dir, "mono/MarianMT+bm25/%s-%s/" % (qlang, dlang))
  elif preranking_model == "unigram":
    assert qlang == dlang
    rerank_dir = os.path.join(prerank_dir, "mono/qlm/%s-%s/" % (qlang, dlang))
  elif preranking_model == "fasttext":
    assert qlang == dlang
    rerank_dir = os.path.join(prerank_dir, "mono/fasttext/IDF-SUM/%s-%s/raw/" % (qlang, dlang))

  # Path to cross-lingual ranking files
  elif preranking_model == "procb":
    rerank_dir = os.path.join(prerank_dir, "xling/clwe/procb/IDF-SUM/%s-%s/" % (qlang, dlang))
  else:
    rerank_dir = os.path.join(prerank_dir, "xling/%s/%s-%s/" % (preranking_model, qlang, dlang))

  logger.info("rerank dir: %s" % rerank_dir)
  assert os.path.exists(rerank_dir) and len(os.listdir(rerank_dir)) > 0, f"Directory empty or does not exist: {rerank_dir}"
  return rerank_dir


def mean_avg_precision(
    query2ranking: Dict[Q_ID, Union[List[DOC_ID], np.array]],
    relass: Dict[Q_ID, List[DOC_ID]],
    **kwargs
) -> MAP_PVAL:
  """
  Evaluates results for queries in terms of Mean Average Precision (MAP). Evaluation gold standard is
  loaded from the relevance assessments.

  :param query2ranking: (actual) ranking for each query
  :param relass: gold standard (expected) ranking for each query
  :return: tuple(MAP, p-value)
  """
  # collect AP values for MAP
  average_precision_values = []
  
  # collect all precision values for significance test
  all_precisions = []
  
  for query_id, ranking in query2ranking.items():
    if query_id in relass:  # len(relevant_docs) > 0:
      relevant_docs = relass[query_id]
      
      # get ranking for j'th query
      is_relevant = [document in relevant_docs for document in ranking]
      ranks_of_relevant_docs = np.where(is_relevant)[0].tolist()
      
      precisions = []
      # +1 because of mismatch betw. one based rank and zero based indexing
      for k, rank in enumerate(ranks_of_relevant_docs, 1):
        precision_at_k = k / (rank + 1)
        precisions.append(precision_at_k)
      all_precisions.extend(precisions)
      
      if len(precisions) == 0:
        print("Warning: query %s without relevant documents in corpus: %s (skipped)" % (query_id, relevant_docs))
      else:
        ap = np.mean(precisions)
        average_precision_values.append(ap)
        
  save_rankings_dir = kwargs.get('save_rankings_dir', None)
  if save_rankings_dir:
    print("saving rankings to %s" % save_rankings_dir)
    os.makedirs(save_rankings_dir, exist_ok=True)
    for qid, ranked_docs in query2ranking.items():
      with open(save_rankings_dir + str(qid) + ".tsv", "w") as f:
        if type(ranked_docs) != list:
          ranked_docs  = ranked_docs.tolist()
        f.writelines([str(did_or_score) + "\n" for did_or_score in ranked_docs])
        
  mean_average_precision = float(np.mean(np.array(average_precision_values)))
  
  if 'save_precision_values_dir' in kwargs:
    tgt_dir = kwargs['save_precision_values_dir']
    os.makedirs(tgt_dir, exist_ok=True)
    with open(os.path.join(tgt_dir, "precision_values.txt"), 'w') as f:
      f.writelines([str(pvalue) + "\n" for pvalue in all_precisions])
      
  # Significance test 
  # against reference model (proc-B)
  file = ""
  if 'load_precision_values_dir' in kwargs:
    file = os.path.join(kwargs['load_precision_values_dir'], "precision_values.txt")
    
  # run signifance test
  pvalue = -1.0
  if os.path.exists(file):
    with open(file, "r") as f:
      reference_precision_values = [float(line.strip()) for line in f.readlines()]
    pvalue = ttest_ind(reference_precision_values, all_precisions)[1]
    
  return mean_average_precision, pvalue


def print_results(durations: List[str], map_results: List[str]):
  """Utility function for logging"""
  logger.info("------------- Mean Average Precision -------------")
  for line in map_results:
    logger.info(line)
  logger.info("------------- Query Latency (ms) -------------")
  for line in durations:
    logger.info(line)
  print("\n------------- Mean Average Precision -------------")
  for line in map_results:
    print(line)
  print("\n------------- Query Latency (ms) -------------")
  for line in durations:
    print(line)


def rerank_and_eval(
    qlang: str,
    dlang: str,
    reranker: CrossEncoder,
    preranker: str,
    prerank_dir: str,
    eval_preranker: bool=True,
    **kwargs
) -> Dict:
  """
  Re-rank and evaluate:
  1. Load for all queries pre-ranking files (qid2topk_rerank, {query-id: [all document-ids ranked])
  2. Load only documents that need to be scored by the re-ranker (documents in the top-k results lists for any query)
  3. Re-rank the top-k documents for each query (scored_doc_ids, doc_scores)
  4. Concatenate re-ranked top-k results with all other documents (qid2reranked, {query-id: [top-k reranked] + [all document-ids ranked][top-k:])
  5. Evaluate pre-ranking (refMAP) and re-ranking (MAP).
  
  result = {
    "MAP": performance of re-ranker, 
    "refMAP": performance of pre-ranker, 
    "duration_ms": milliseconds/query, 
    "pvalue": pvalue
  }
  
  :param qlang: query language
  :param dlang: document language
  :param reranker: instance of @CrossEncoder (sentence-transformers library), used to re-rank initial ranking of :param preranker
  :param preranker: any of the following: bm25, fbmt+bm25, marianmt+bm25, unigram, procb, distil_mbert
  :param prerank_dir: Directory containing prerankings for all language pairs and models, each file corresponds to the full ranking of a single query
  :param eval_preranker: additionally compute Mean Average Precision of pre-ranker only ("refMAP")
  :param kwargs: (optional) save_precision_values_dir: str, 'load_precision_values_dir: str, 'run_split_adapters: bool 
  :return: results summary in terms of efficiency (query latency, milliseconds) and effectiveness (MAP)
  """
  if qlang in ["sw", "so"]:
    path_query_translations = kwargs.get("path_query_translations", None)
    assert path_query_translations, "Path for query translation files not specified, run bm25_eval.py first."
    rerank_dir = _get_rerank_dir(dlang, preranker, qlang, prerank_dir)
    doc_ids, documents, _, _, relass, qid2topk_rerank = load_clef_rerank(
      qlang="en", dlang=dlang, rerank_dir=rerank_dir, topk=topk)
    
    mt_system = preranker.split("+")[0]
    with open(os.path.join(path_query_translations, f"{qlang}_to_en_translated_queries_{mt_system}.pkl"), "rb") as f:
      query_ids, queries = pickle.load(f)
    query_ids = [int(_id) for _id in query_ids]
    
  else:
    rerank_dir = _get_rerank_dir(dlang, preranker, qlang, prerank_dir)
    doc_ids, documents, queries, query_ids, relass, qid2topk_rerank = load_clef_rerank(
      qlang=qlang, dlang=dlang, rerank_dir=rerank_dir, topk=topk)
    
  qid2reranked = {}
  durations = []
  with torch.no_grad():
    for qid, query in tqdm(zip(query_ids, queries), total=len(queries)):
      if qid in relass:
        doc_scores = []
        scored_doc_ids = []
        
        if kwargs.get("run_split_adapters", False):
          import transformers.adapters.composition as ac
          task_adapter_name = kwargs["task_adapter_name"]
          # determine split index IDX by feeding a dummy document, all tokens before IDX are fed to the query language 
          # adapter, all tokens after IDX are fed to the document language adapter
          split_index = reranker.tokenizer(
            [[query, "Dummy document " * 50] for _ in range(10)], padding=True, truncation='longest_first', max_length=512
          )["token_type_ids"][0].index(1)
          reranker.model.active_adapters = ac.Stack(ac.Split(qlang, dlang, split_index=split_index), task_adapter_name)
        
        start = time.perf_counter_ns()
        
        docs_batch = []
        for did in qid2topk_rerank[qid]:
          docs_batch.append(documents[doc_ids.index(did)])
          scored_doc_ids.append(did)
          if len(docs_batch) % batch_size == 0 and len(docs_batch) > 0:
            cross_inp = [[query, doc] for doc in docs_batch]
            
            # Cross-Encoder that predict more than 1 score, we use the last and apply softmax
            if reranker.config.num_labels > 1:
              scores = reranker.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
            else:
              scores = reranker.predict(cross_inp).tolist()
              
            doc_scores.extend(scores)
            docs_batch = []
            
          if len(scored_doc_ids) == topk:
            break
            
        tmp_ranking = [(tmpdid, score) for tmpdid, score in zip(scored_doc_ids, doc_scores)]
        tmp_ranking = sorted(tmp_ranking, key=lambda elem: -elem[1])
        ranking = [elem[0] for elem in tmp_ranking] + qid2topk_rerank[qid][topk:]
        
        duration = time.perf_counter_ns() - start
        durations.append((duration, len(cross_inp)))
        qid2reranked[qid] = ranking
        
  # Compute query latency
  # Skip first call of .predict() as it takes much longer than usual (not representative)
  durations = durations[1:]
  nanoseconds_per_query = sum([duration/size for duration, size in durations]) / len(durations)
  milliseconds_per_query = round(nanoseconds_per_query / 1000000, 2)
  
  # Compute statistical signifance
  MAP, pvalue = mean_avg_precision(qid2reranked, relass, **kwargs)
  refMAP = mean_avg_precision(qid2topk_rerank, relass)[0] if eval_preranker else -1
  logger.info(f"{qlang}->{dlang}\t"
              f"reranker (MAP): {str(MAP)}\t"
              f"preranker (MAP): {refMAP}\t"
              f"duration/query (ms): {milliseconds_per_query}\t"
              f"pvalue: {pvalue}")
  return {"MAP": MAP, "refMAP": refMAP, "duration_ms": milliseconds_per_query, "pvalue": pvalue}
