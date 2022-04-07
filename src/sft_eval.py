import sys
import os
import argparse
import pprint
import logging

from sentence_transformers import CrossEncoder
from sft import SFT
from helper.evaluate import add_filehandler, rerank_and_eval, map2str, print_results
from helper.config import max_seq_len, get_language_pairs, lang_rf, get_preranker


# Mandatory
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Location of (downloaded/trained) Sparse Finetuning Masks.", required=True)
parser.add_argument("--mode", type=str, choices=['lowres', 'mono', 'clir'], help="clir: reproduce Table 1, lowres: reproduce Table 2; mono: reproduce Table 3.", required=True)
parser.add_argument("--prerank_dir", type=str, help="Location of preranking files", required=True)
# Optional 
parser.add_argument("--task_rf", choices=['1', '2', '4', '8', '16', '32'], help="Task reduction factors to evaluate.", nargs='+', default=['2'])
parser.add_argument("--language_configs", choices=['qlang', 'dlang', 'both'], type=str, help="Query language SFTM (qlang), document language SFTM (dlang), both SFTMs (both)", nargs='+', default=['dlang'])
parser.add_argument("--gpu", type=str, help="Value for CUDA_VISIBLE_DEVICES.", default="", required=False)
parser.add_argument("--ttest_reference", type=str, help="Location of folder with precision files.", required=False, default="")
parser.add_argument("--path_query_translations", type=str, help="Only required for --mode lowres, path to query translations (output of bm25_eval.py)", required=False)
args = parser.parse_args()

logging.basicConfig(
  format="%(asctime)s [%(levelname)s]: %(message)s",
  datefmt="%d.%m.%Y %H:%M:%S",
  handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('\n' + pprint.pformat(vars(args)))

SFT_DIR = args.model_dir
PRERANK_DIR = args.prerank_dir
REF_PRECISION_VALS_DIR = args.ttest_reference
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
lang_pairs = get_language_pairs(args.mode, args.path_query_translations)
task_reduction_factors = args.task_rf
language_configs = args.language_configs
assert os.path.exists(PRERANK_DIR)
logger.info('\n' + pprint.pformat(vars(args)))


def evaluate_sfts():
  add_filehandler(os.path.join(SFT_DIR, f"evaluate_sfts.txt"))
  logger.info(f"loading precomputed average precision (AP) values for t-test from {REF_PRECISION_VALS_DIR}")

  reranker = CrossEncoder('bert-base-multilingual-uncased', max_length=max_seq_len)
  map_results = []
  duration_results = []
  for task_rf in task_reduction_factors:
    for steps in ['625k']: # '100k'
      
      # Apply Ranking Mask (RM)
      RM_path = os.path.join(SFT_DIR, f"ir_{steps}/rf_{lang_rf}_{task_rf}/checkpoint-625000")
      ranking_sft = SFT(RM_path)
      ranking_sft.apply(reranker.model, with_abs=True)

      for language_config in language_configs: 
        langpair2map = {}
        langpair2duration = {}
        for qlang, dlang in lang_pairs:
          logger.info("------------------")
          active_lang_sfts = []
          
          # Apply Language Masks (LM)
          if language_config == 'qlang' or language_config == 'both':
            # For Swahili/Somali to English ({swc,so}->en) we use FacebookMT to translate queries to EN and prerank
            # with BM25 ('fbmt+bm25'), so we need to the English SFT here.
            sft_language = qlang if qlang not in ('sw', 'so') else 'en'
            LM_path = os.path.join(SFT_DIR, f"mlm/rf_{lang_rf}/{sft_language}")
            sft = SFT(LM_path)
            sft.apply(reranker.model, with_abs=False)
            active_lang_sfts.append(sft)
          if language_config == 'dlang' or language_config == 'both':
            LM_path = os.path.join(SFT_DIR, f"mlm/rf_{lang_rf}/{dlang}")
            sft = SFT(LM_path)
            sft.apply(reranker.model, with_abs=False)
            active_lang_sfts.append(sft)
          
          preranker_model = get_preranker(qlang, dlang)
          load_precision_values_dir = os.path.join(
            REF_PRECISION_VALS_DIR, f"{qlang}-{dlang}_preranker={preranker_model}"
          )
          save_precision_values_dir = os.path.join(
            RM_path, f"precision_values/{qlang}-{dlang}_setting={language_config}_preranker={preranker_model}"
          )
          
          logger.info(f"Load precision values from: {load_precision_values_dir}")
          logger.info(f"Save precision values to: {save_precision_values_dir}")
          logger.info(f"preranker: {preranker_model}")
          eval_result = rerank_and_eval(
            qlang=qlang,
            dlang=dlang,
            reranker=reranker,
            preranker=preranker_model,
            prerank_dir=PRERANK_DIR,
            load_precision_values_dir=load_precision_values_dir,
            save_precision_values_dir=save_precision_values_dir,
            path_query_translations=args.path_query_translations
          )
          langpair2map[qlang + dlang] = map2str(eval_map=eval_result["MAP"], pvalue=eval_result["pvalue"])
          langpair2duration[qlang + dlang] = str(eval_result["duration_ms"])
          logger.info(f"{qlang}->{dlang}\tsetting: {language_config}\tsteps: {steps}\ttask-rf: {task_rf}\tlang-rf: 2")
          
          for lang_sft in active_lang_sfts:
            lang_sft.revert(reranker.model)
            
        if not map_results:
          header = "task_rf\tcheckpoint\tsetting\t"
          map_results.append(header + "\t".join(langpair2map.keys()))
          duration_results.append(header)

        desc = [str(task_rf), steps, language_config]
        map_results.append("\t".join(desc + [langpair2map[lp] for lp in langpair2map.keys()]))
        duration_results.append("\t".join(desc + [langpair2duration[lp] for lp in langpair2map.keys()]))
        
      ranking_sft.revert(reranker.model)
      
  print_results(durations=duration_results, map_results=map_results)
  
  return map_results, duration_results


def main():
  evaluate_sfts()


if __name__ == '__main__':
    main()
