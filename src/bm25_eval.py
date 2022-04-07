"""
This script requires a set up pyserini index and runs BM25 on maybe translated queries

Dataset source for Somali and Swahili CLEF queries: 
https://ciir.cs.umass.edu/ictir19_simulate_low_resource
"""
import sys
import os
import pickle
import argparse
import logging
import pprint

from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from easynmt import EasyNMT
from tqdm import tqdm
from clef_dataloader import load_queries, load_relevance_assessments
from helper.evaluate import mean_avg_precision
from helper.config import all_lang_pairs


parser = argparse.ArgumentParser()
parser.add_argument("--index_dir", type=str, help="Location of Lucene indexes, one for each language (INDEX_HOME in scripts/index.sh).")
parser.add_argument("--output_dir", type=str, help="Output location of pre-ranking files (input for re-ranking) and query translation files.")
parser.add_argument("--lang_pairs", action='append', nargs='+', help="One or more space-separated language pairs, e.g. 'ende defi'")
parser.add_argument("--save_rankings", action='store_true', default=False, help="Whether to save output rankings for reranking.")
args = parser.parse_args()

logging.basicConfig(
  format="%(asctime)s [%(levelname)s]: %(message)s",
  datefmt="%d.%m.%Y %H:%M:%S",
  handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('\n' + pprint.pformat(vars(args)))

# Evaluate on CLEF-2003 benchmark
year = "2003"

# Supported language pairs
lang_pairs = [(lp[:2], lp[2:]) for lp in args.lang_pairs[0]]
assert all(lp in all_lang_pairs for lp in lang_pairs), "Language-pair not supported."

monolingual_save_dir = os.path.join(args.output_dir, f"preranking/mono/bm25")
os.makedirs(monolingual_save_dir, exist_ok=True)

xling_save_dir =  os.path.join(args.output_dir, f"preranking/xling/fbnmt+bm25")
os.makedirs(xling_save_dir, exist_ok=True)
logger.info(f"Saving ranking files to {monolingual_save_dir} / {xling_save_dir}")

translations_dir = os.path.join(args.output_dir, "translated_queries/")
os.makedirs(translations_dir, exist_ok=True)
logger.info(f"Saving translation files to {translations_dir}")

translate = None
# multilingual translation model needs to be loaded only once and can be applied to all language pairs above
if any(query_language != document_language for query_language, document_language in lang_pairs):
  model = EasyNMT('m2m_100_418M')

for qlang, dlang in lang_pairs:
  logger.info(f"Running {qlang}->{dlang}")
  index_dir = os.path.join(args.index_dir, dlang)
  index_reader = IndexReader(index_dir)
  n_docs = 100000 

  searcher = SimpleSearcher(index_dir)
  searcher.set_language(dlang)
  searcher.set_bm25()

  relass = load_relevance_assessments(language=dlang, year=year)
  query_ids, queries = load_queries(language=qlang, year=year)

  if qlang != dlang:
    translations_file = os.path.join(translations_dir, f"{qlang}_to_{dlang}_translated_queries_fbnmt.pkl")
    if os.path.exists(translations_file):
      logger.info(f"Loading translations from {translations_file}")
      with open(translations_file, "rb") as f:
        query_ids, queries = pickle.load(f)
    else:
      logger.info("Running translation")
      translate = lambda input_queries: [model.translate(q, source_lang=qlang,target_lang=dlang) for q in tqdm(input_queries)]
      queries = translate(queries)
      logger.info(f"Saving translations to {translations_file}")
      with open(translations_file, "wb") as f:
        pickle.dump((query_ids, queries), f)
  
  # Retrieve top documents from lucene index
  batch_size = 5
  n_queries = 60
  threads = os.cpu_count() // 2
  lucene_results = {}
  for i in tqdm(list(range(0, n_queries, batch_size))):
    batch_ranking = searcher.batch_search(
      queries=queries[i:i + batch_size], 
      qids=[str(qid) for qid in query_ids[i:i + batch_size]], 
      k=n_docs, 
      threads=threads
    )
    for qid, ranking in batch_ranking.items():
      lucene_results[int(qid)] = [result.docid for result in ranking]

  # searcher.batch_search(...) does not rank the full corpus but the top-100k, append remaining document ids.
  # This has an insignificant effect on mean average precision.
  tmp =  {}
  for qid, relevant_docs in relass.items():
    ranking = lucene_results[qid]
    missing = [rel_doc for rel_doc in relevant_docs if rel_doc not in ranking]
    tmp[qid] = lucene_results[qid] + missing
  lucene_results = {int(k): v for k, v in tmp.items()}

  # Evaluate retrieval result
  mean_ap, _ = mean_avg_precision(query2ranking=lucene_results, relass=relass)
  logger.info(f"{qlang}-{dlang}:\t {mean_ap}")

  # Saving output files for re-ranking
  if args.save_rankings:
    save_rankings_dir = monolingual_save_dir if qlang == dlang else xling_save_dir
    lang_pair_output_dir = os.path.join(save_rankings_dir, f"{qlang}-{dlang}/")
    os.makedirs(lang_pair_output_dir, exist_ok=True)
    logger.info(f"Saving rankings to {lang_pair_output_dir}")
    for qid, ranking in lucene_results.items():
      with open(os.path.join(lang_pair_output_dir, f"{qid}.tsv"), "w") as f:
        f.writelines([f"{docid}\n" for docid in ranking])
