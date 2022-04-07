import os
import argparse
import logging

from sentence_transformers import CrossEncoder
from helper.evaluate import add_filehandler, logger, rerank_and_eval, map2str, print_results
from helper.config import crosslingual_lang_pairs, monolingual_lang_pairs, low_res_lang_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Location of monoBERT checkpoint (output of run_monoBERT_retrieval.sh)", required=True)
parser.add_argument("--gpu", type=str, help="Value for CUDA_VISIBLE_DEVICES.", default="", required=False)
parser.add_argument("--mode", type=str, choices=["lowres", "mono", "clir"], help="clir: reproduce Table 1, lowres: reproduce Table 2; mono: reproduce Table 3.", required=True)
parser.add_argument("--path_query_translations", type=str, help="Only required for --mode lowres, path to query translations (output of bm25_eval.py)", required=False)
parser.add_argument("--prerank_dir", type=str, help="Location of preranking files", required=True)
parser.add_argument("--path_logging", type=str, default="")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
BASELINE_DIR = args.model_dir
assert os.path.exists(BASELINE_DIR)

# Directory for reference precision values, used for statistical tests
REF_PRECISION_VALS_DIR = os.path.join(BASELINE_DIR, "precision_values")
os.makedirs(REF_PRECISION_VALS_DIR, exist_ok=True)

if args.mode == "lowres":
  assert os.path.exists(args.path_query_translations)
  assert len(os.listdir(args.path_query_translations)) > 0

def _evaluate_single_baseline(own_model, mode):
  if args.path_logging:
    add_filehandler("evaluate_baseline.txt")
    logger.setLevel(logging.INFO)
  logger.info(f"Save average precision values to {REF_PRECISION_VALS_DIR}")
  
  if own_model:
    model_name_path = BASELINE_DIR
  else:
    model_name_path = 'amberoad/bert-multilingual-passage-reranking-msmarco'
  
  logger.info(model_name_path)
  reranker = CrossEncoder(model_name_path, max_length=512)
  
  if mode == "mono": 
    prerankers = ["bm25"] # "unigram", "fasttext"
    lang_pairs = monolingual_lang_pairs
    
  elif mode == "lowres":
    prerankers = ["fbnmt+bm25"] # "marianmt+bm25"
    lang_pairs = low_res_lang_pairs
    
  else:
    assert mode == "clir"
    prerankers = ["distil_mbert"] # "procb", "distil_xlmr", "distil_muse", "muse", "labse", "laser"
    lang_pairs = crosslingual_lang_pairs
  
  header = "\t".join([l1 + l2 for l1, l2 in lang_pairs])
  map_results = [header]
  duration_results = [header]
  
  for preranker_str in prerankers:
    preranker_model = preranker_str
    
    langpair2map = {}
    langpair2duration = {}
    for qlang, dlang in lang_pairs:
      save_precision_values_dir= os.path.join(REF_PRECISION_VALS_DIR, f"{qlang}-{dlang}_preranker={preranker_str}")
      eval_result = rerank_and_eval(
        qlang=qlang,
        dlang=dlang,
        reranker=reranker,
        preranker=preranker_model,
        prerank_dir=args.prerank_dir,
        save_precision_values_dir=save_precision_values_dir,
        path_query_translations=args.path_query_translations
      )
      langpair2map[qlang + dlang] = map2str(eval_map=eval_result["MAP"], pvalue=eval_result["pvalue"])
      langpair2duration[qlang + dlang] = str(eval_result["duration_ms"])
      
    map_results.append("\t".join([preranker_str] + [langpair2map[q+d] for q,d in lang_pairs]))
    duration_results.append("\t".join([preranker_str] + [langpair2duration[q+d] for q, d in lang_pairs]))
    
  logger.info(model_name_path)
  print_results(durations=duration_results, map_results=map_results)
  
  return map_results, duration_results


def evaluate_baselines():
  all_results = [] 
  all_durations = [] 
  for mode in [args.mode]: 
    map_results, duration_results = _evaluate_single_baseline(own_model=True, mode=mode)
    all_results.extend([mode + "\t" + line for line in map_results[1:]])
    all_durations.extend([mode + "\t" + line for line in duration_results[1:]])
  print_results(durations=all_durations, map_results=all_results)


def main():
  evaluate_baselines()


if __name__ == '__main__':
    main()
