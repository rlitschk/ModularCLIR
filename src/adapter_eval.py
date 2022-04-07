import os
import sys
import argparse
import logging
import pprint

from pprint import pformat
from sentence_transformers import CrossEncoder
from transformers.adapters import composition as ac
from helper.evaluate import rerank_and_eval
from helper.evaluate import add_filehandler
from helper.evaluate import map2str
from helper.evaluate import print_results
from helper.config import max_seq_len, lang_rf, get_preranker, get_language_pairs

# Mandatory
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Location of (downloaded/trained) adapters.", required=True)
parser.add_argument("--mode", type=str, choices=['lowres', 'mono', 'clir'], help="clir: reproduce Table 1, lowres: reproduce Table 2; mono: reproduce Table 3.", required=True)
parser.add_argument("--prerank_dir", type=str, help="Location of preranking files", required=True)
# Optional 
parser.add_argument("--task_rf", choices=['1', '2', '4', '8', '16', '32'], help="Task reduction factors to evaluate.", nargs='+', default=['16'])
parser.add_argument("--language_configs", choices=['qlang', 'dlang', 'split', '+ra+la-inv', '+ra-la-inv'], type=str, help="Query language adapter (qlang), document language adapter (dlang), Split Adapter (split), document language adapter w/o invertible adapters (+ra+la-inv), document language adapter w/o language and invertible adapters (+ra-la-inv)", nargs='+', default=['dlang'])
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


ADAPTER_DIR = args.model_dir
PRERANK_DIR = args.prerank_dir
REF_PRECISION_VALS_DIR = args.ttest_reference
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lang_pairs = get_language_pairs(args.mode, args.path_query_translations)


# Input validation
if 'split' in args.language_configs and args.mode == 'lowres':
  logger.warning("Split adapters currently not supported for low-resource languages, ignoring 'split' parameter.")
  while 'split' in args.language_configs:
    args.language_configs.pop(args.language_configs.index('split'))
assert os.path.exists(PRERANK_DIR)


def _evaluate_adapter_setting(eval_adapterdrop=False, ablation_config=None, name=""):
  fName = "evaluate_AdapterDrop.txt" if eval_adapterdrop else "evaluate_adapter_ablation_tmp.txt"
  add_filehandler(os.path.join(ADAPTER_DIR, fName))
  logger.info(f"loading precomputed average precision (AP) values for t-test from {REF_PRECISION_VALS_DIR}")
  config_to_str = lambda sl_config: f"{sl_config[0]}-{sl_config[-1]}" if type(sl_config) == list else str(sl_config)
  
  if eval_adapterdrop:
    apply_task_adapter = True
    apply_lang_adapter = True
    apply_invertible_adapters = True
    # AdapterDrop ablation settings (skip_layer_configs): ['None', '1-2', '1-4', '1-6', '1-8', '1-10', '1-12']
    # Example: https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/adapterdrop/drop_at_inference.py
    skip_layer_configs = [list(range(1, k)) for k in range(1, 12+2, 2)]
    skip_layer_configs[0] = None
    task_reduction_factors = ['1', '2', '4', '8', '16', '32']
    language_config = 'dlang'
    # 'pfeiffer' is trained without invertible adapters, 'pfeiffer+inv' trained with invertible adapters
    # We observe that training with 'pfeiffer+inv', i.e., with invertible adapters during MLM pretraining of language
    # adapters, and then turning them off during inference (apply_invertible_adapters = False) improves results.
    adapter_config = 'pfeiffer+inv'
  else:
    if ablation_config:
      skip_layer_configs = ablation_config['skip_layer_configs']
      apply_task_adapter = ablation_config['apply_task_adapter']
      apply_lang_adapter = ablation_config['apply_lang_adapter']
      apply_invertible_adapters = ablation_config['apply_invertible_adapters']
      language_config = ablation_config['setting']
      task_reduction_factors = ablation_config['task_reduction_factors']
      adapter_config = ablation_config['config']
    else:
      skip_layer_configs = [None]
      apply_task_adapter = True
      apply_lang_adapter = True
      apply_invertible_adapters = False
      language_config = 'dlang'
      task_reduction_factors = ['16']
      adapter_config = 'pfeiffer+inv'
  
  logger.info(f"task adapters: {apply_task_adapter}"
              f"\tlanguage adapters: {apply_lang_adapter}"
              f"\tinvertible adapters:{apply_invertible_adapters}"
              f"\tskip layers: {[config_to_str(c) for c in skip_layer_configs]}"
              f"\tadapter-config: {adapter_config}")
  logger.info(f"Setting: {language_config}\tlang_reduction_factor: {lang_rf}\ttask reduction factor: {task_reduction_factors}")

  task_rf2results = {}
  task_rf2durations = {}

  for task_rf in task_reduction_factors:
    reranker = CrossEncoder('bert-base-multilingual-uncased', max_length=max_seq_len)

    # Load Ranking Adapter
    if apply_task_adapter:
      if adapter_config == 'pfeiffer+inv':
        RA_path = os.path.join(ADAPTER_DIR, f"ir/rf_{lang_rf}_{task_rf}/checkpoint-625000/retrieval")
      else:
        RA_path = os.path.join(ADAPTER_DIR, f"ir_-INV/rf_{lang_rf}_{task_rf}/checkpoint-625000/retrieval")
      reranker.model.load_adapter(
        RA_path,
        load_as='ir',
        with_head=True
      )

    langpair2skip_layers2map = {}
    langpair2skip_layers2duration = {}
    for qlang, dlang in lang_pairs:
      preranker_model = get_preranker(qlang, dlang)

      # Load Language Adapter
      if apply_lang_adapter:
        # Load Language Adapter, for so-en and sw-en we load translated queries, hence load en adapter
        # LA_lang = dlang if language_config == 'dlang' or qlang in ('so', 'sw') else qlang
        LA_lang = dlang if language_config == 'dlang' or qlang in ('so', 'sw') else qlang
        LA_path = os.path.join(ADAPTER_DIR, f"mlm/rf_{lang_rf}/{LA_lang}/checkpoint-225000/mlm_adapter")
        # maybe adjust for architecture where invertible adapters are turned off already during training (not part of paper)
        if adapter_config == 'pfeiffer':
          LA_path = LA_path.replace("mlm/", "mlm_-INV/")

        reranker.model.load_adapter(
          LA_path,
          load_as="mlm",
          with_head=False
        )
        if not apply_invertible_adapters:
          reranker.model.bert.delete_invertible_adapter('mlm')

      skip_layers2map = {}
      skip_layers2duration = {}
      for skip_layers in skip_layer_configs:
        skip_layers_str = config_to_str(skip_layers)
        logger.info(f"Skip layers: {skip_layers_str}\t{qlang}->{dlang}\ttask_rf: {task_rf}\tlang_rf: {lang_rf}")

        # Set activate Adapters
        if apply_task_adapter and apply_lang_adapter:
          reranker.model.set_active_adapters(ac.Stack("mlm", "ir"), skip_layers=skip_layers)
        elif apply_task_adapter and not apply_lang_adapter:
          reranker.model.set_active_adapters("ir", skip_layers=skip_layers)
        elif not apply_task_adapter and apply_lang_adapter:
          raise NotImplementedError

        save_precision_values_dir = os.path.join(
          ADAPTER_DIR, 
          f"ir/rf_2_{task_rf}/precision_values"
          f"{qlang}-{dlang}_+LANG-INV_setting={language_config}_preranker={preranker_model}"
        )
        logger.info(f"saving precision values to: {save_precision_values_dir}")

        load_precision_values_dir = os.path.join(REF_PRECISION_VALS_DIR, f"{qlang}-{dlang}_preranker={preranker_model}")
        logger.info(f"loading precision values from: {load_precision_values_dir}")

        # Re-rank and evaluate
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
        skip_layers2duration[skip_layers_str] = str(eval_result["duration_ms"])
        skip_layers2map[skip_layers_str] = map2str(eval_map=eval_result["MAP"], pvalue=eval_result["pvalue"])

      langpair2skip_layers2map[qlang + dlang] = skip_layers2map
      langpair2skip_layers2duration[qlang + dlang] = skip_layers2duration

    task_rf2results[task_rf] = langpair2skip_layers2map
    task_rf2durations[task_rf] = langpair2skip_layers2duration
    
    # Move adapters to cpu to avoid memory issues
    reranker.model.cpu()

  header = "task_rf\tskip_layers\t" + name if name else "" + "\t".join([f"{q}{d}" for q, d in lang_pairs])
  map_results = [header]
  duration_results = [header]
  
  name_str = f"'{name}\t" if name else ""
  for task_rf in task_rf2results.keys():
    langpair2skip_layers2map = task_rf2results[task_rf]
    langpair2skip_layers2duration = task_rf2durations[task_rf]
    tmp_lang_pairs = list(langpair2skip_layers2duration.keys())
    skip_layer_configs = langpair2skip_layers2map[tmp_lang_pairs[0]]
    for skip_layers in skip_layer_configs:
      skip_layers_str = config_to_str(skip_layers)
      map_results.append(
        f"{task_rf}\t{skip_layers_str}\t{name_str}" + "\t".join(
          [langpair2skip_layers2map[lp][skip_layers_str] for lp in tmp_lang_pairs]
        )
      )
      duration_results.append(
        f"{task_rf}\t{skip_layers_str}\t{name_str}" + "\t".join(
          [langpair2skip_layers2duration[lp][skip_layers_str] for lp in tmp_lang_pairs]
        )
      )
      
  return map_results, duration_results


def evaluate_adapters():
  fName = f"evaluate_adapters.txt"
  add_filehandler(os.path.join(ADAPTER_DIR, fName))
  logger.info(f"loading precomputed average precision (AP) values for t-test from {REF_PRECISION_VALS_DIR}")
  
  reranker = CrossEncoder('bert-base-multilingual-uncased', max_length=max_seq_len)
  map_results = []
  duration_results = []
  checkpoints = [625000] # 100000
  task_reduction_factors = args.task_rf
  language_configs = args.language_configs
  
  for task_rf in task_reduction_factors:
    for checkpoint in checkpoints:
      for language_config in language_configs:
        # Load Ranking Adapter
        RA_name = f"RA_rf={task_rf}"
        RA_path = os.path.join(ADAPTER_DIR, f"ir/rf_{lang_rf}_{task_rf}/checkpoint-625000/retrieval")
        
        reranker.model.load_adapter(
          RA_path,
          load_as=RA_name,
          with_head=True
        )
        langpair2map = {}
        langpair2duration = {}
        logger.info(f"Evaluating language pairs: {lang_pairs}")
        for qlang, dlang in lang_pairs:
          preranker_model = get_preranker(qlang, dlang)
          
          # Load Language Adapter, for so-en and sw-en we load translated queries, hence load en adapter
          LA_lang = dlang if language_config == 'dlang' or qlang in ('so', 'sw') else qlang 
          LA_path = os.path.join(ADAPTER_DIR, f"mlm/rf_{lang_rf}/{LA_lang}/checkpoint-225000/mlm_adapter") 
          LA_name = f"LA_rf={lang_rf}"
          logger.info(f"Loading language adapter {LA_path}")
          
          reranker.model.load_adapter(
            LA_path,
            load_as=LA_name,
            with_head=False
          )
          reranker.model.set_active_adapters(ac.Stack(LA_name, RA_name))
          logger.info("language adapter:\t" + LA_path)
          logger.info("task adapter:\t" + RA_path)
          logger.info("setting:\t" + language_config)

          path_save_precision_values_dir = RA_path.replace("retrieval", "precision_values")
          load_precision_values_dir = os.path.join(REF_PRECISION_VALS_DIR, f"{qlang}-{dlang}_preranker={preranker_model}")
          save_precision_values_dir=os.path.join(
            path_save_precision_values_dir, f"{qlang}-{dlang}_setting={language_config}_preranker={preranker_model}"
          )
          logger.info(f"saving precision values to: {save_precision_values_dir}")
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
          logger.info("------")

          langpair2map[qlang + dlang] = map2str(eval_map=eval_result['MAP'], pvalue=eval_result['pvalue'])
          langpair2duration[qlang + dlang] = str(eval_result['duration_ms'])

        if not map_results:
          header = "task_rf\tcheckpoint\tlanguage_config\t" + "\t".join(langpair2map.keys())
          map_results.append(header)
          duration_results.append(header)

        desc = [str(task_rf), str(checkpoint), language_config]
        map_results.append("\t".join(desc + [langpair2map[lp] for lp in langpair2map.keys()]))
        duration_results.append("\t".join(desc + [langpair2duration[lp] for lp in langpair2map.keys()]))
  
  return map_results, duration_results


def evaluate_split_adapters():
  fName = "evaluate_split-adapters.txt"
  add_filehandler(os.path.join(ADAPTER_DIR, fName))
  
  # Best setting for split-adapter: [LA_rf=2, RA_rf=32]
  language_config = 'split-adapter'
  task_reduction_factors = args.task_rf
  checkpoints = [625000] # 100000
  
  logger.info(f"Setting: {language_config}\t"
              f"lang_reduction_factor: {lang_rf}\t"
              f"task reduction factor: {task_reduction_factors}")
  
  task_rf2results = {}
  task_rf2durations = {}
  for task_rf in task_reduction_factors:
    
    ckpt2lp2maps = {}
    ckpt2lp2durations = {}
    for checkpoint in checkpoints:
      
      reranker = CrossEncoder("bert-base-multilingual-uncased", max_length=max_seq_len)
      TA_path = os.path.join(ADAPTER_DIR, f"ir/rf_{lang_rf}_{task_rf}/checkpoint-625000/retrieval")
      task_adapter_name = 'ir'
      reranker.model.load_adapter(
        TA_path,
        load_as=task_adapter_name,
        with_head=True
      )
      
      loaded_adapters = set()
      langpair2map = {}
      langpair2duration = {}
      for qlang, dlang in lang_pairs:
        preranker_model = get_preranker(qlang, dlang)
        
        logger.info("------------------")
        if qlang not in loaded_adapters:
          reranker.model.load_adapter(
            os.path.join(ADAPTER_DIR, f"mlm/rf_{lang_rf}/{qlang}/checkpoint-225000/mlm_adapter"),
            load_as=qlang,
            with_head=False
          )
          loaded_adapters.add(qlang)
        if dlang not in loaded_adapters:
          reranker.model.load_adapter(
            os.path.join(ADAPTER_DIR, f"mlm/rf_{lang_rf}/{dlang}/checkpoint-225000/mlm_adapter"),
            load_as=dlang,
            with_head=False
          )
          loaded_adapters.add(dlang)
        
        load_precision_values_dir = REF_PRECISION_VALS_DIR + f"{qlang}-{dlang}_preranker={preranker_model}"
        save_precision_values_dir = os.path.join(
          ADAPTER_DIR, 
          f"ir_625k/rf_2_{task_rf}/precision_values", 
          f"{qlang}-{dlang}_setting={language_config}_preranker={preranker_model}"
        ) 
        logger.info(f"saving precision values to: {save_precision_values_dir}")

        eval_result = rerank_and_eval(
          qlang=qlang,
          dlang=dlang,
          reranker=reranker,
          preranker=preranker_model,
          prerank_dir=PRERANK_DIR,
          task_adapter_name=task_adapter_name,
          run_split_adapters=True,
          load_precision_values_dir=load_precision_values_dir,
          save_precision_values_dir=save_precision_values_dir,
          # Split adapters currently not supported for low-resource languages 
          # path_query_translations=args.path_query_translations 
        )
        logger.info(f"{qlang}->{dlang}\tckpt: {checkpoint}\ttask-rf: {task_rf}\tlang-rf: {lang_rf}")
        langpair2map[qlang + dlang] = map2str(eval_map=eval_result['MAP'], pvalue=eval_result['pvalue'])
        langpair2duration[qlang + dlang] = str(eval_result["duration_ms"])

      ckpt2lp2maps[checkpoint] = langpair2map
      ckpt2lp2durations[checkpoint] = langpair2duration
      reranker.model.cpu()
      
    task_rf2results[task_rf] = ckpt2lp2maps
    task_rf2durations[task_rf] = ckpt2lp2durations
    
  header = "\t".join(['task_rf', 'checkpoint', 'language_config'] + [f"{q}{d}" for q, d in lang_pairs])
  map_results = [header]
  duration_results = [header]
  for task_rf, ckpt2results in task_rf2results.items():
    for ckpt, lp2map in ckpt2results.items():
      config = f"{task_rf}\t{ckpt}\tsplit\t"
      map_results.append(config + "\t".join([lp2map[qlang+dlang] for qlang, dlang in lang_pairs]))
      duration_results.append(config + "\t".join([task_rf2durations[task_rf][ckpt][qlang+dlang] for qlang, dlang in lang_pairs]))
      
  return map_results, duration_results


def evaluate_adapter_ablation(selected_configs):
  base_config = {
    # static parameters of best configuration
    "setting": "dlang",
    "skip_layer_configs": [None],
    "task_reduction_factors": ['16'],
    "apply_task_adapter": True
  }
  
  all_configs = {
    "+RA+LA": {
      # base setup: invertible and language adapters enabled during training and inference (+RA +LA), same as just applying dlang
      "apply_lang_adapter": True,
      "apply_invertible_adapters": True,
      "config": "pfeiffer+inv",
      **base_config
    }, "+RA-LA-INV": {
      # turn off language adapters and invertible adapters during inference (+RA -LA -INV)
      "apply_lang_adapter": False,
      "apply_invertible_adapters": False,
      "config": "pfeiffer+inv",
      **base_config
    }, "+RA+LA-INV": {
      # turn off invertible adapters during inference (+RA +LA -INV)
      "apply_lang_adapter": True,
      "apply_invertible_adapters": False,
      "config": "pfeiffer+inv",
      **base_config
    },
    # The following configs are the same as above, but invertible adapters are turned off already during training
    # "[pfeiffer]": {
    #   # base setup: language adapters enabled during training and inference
    #   "apply_lang_adapter": True,
    #   "apply_invertible_adapters": True,
    #   "config": "pfeiffer",
    #   **base_config
    # },"[pfeiffer]-LANG-INV": {
    #   # turn off language adapters and invertible adapters during inference and training
    #   "apply_lang_adapter": False,
    #   "apply_invertible_adapters": False,
    #   "config": "pfeiffer",
    #   **base_config
    # }, "[pfeiffer]-INV": {
    #   # turn off invertible adapters during inference and training
    #   "apply_lang_adapter": True,
    #   "apply_invertible_adapters": False,
    #   "config": "pfeiffer",
    #   **base_config
    # }
  }
  
  all_results = []
  all_durations = []
  for name in selected_configs:
    name = name.upper()
    config = all_configs[name]
    logger.info(f"Running {name}")
    logger.info(pformat(config, indent=4))
    map_results, duration_results = _evaluate_adapter_setting(
      eval_adapterdrop=False, ablation_config=config, name=name
    )
    k = 0 if not all_results else 1 # maybe remove header
    all_results.extend(map_results[k:])
    all_durations.extend(duration_results[k:])
    logger.info(f"Done with {name}")
    logger.info(f"---------")
  
  return all_results, all_durations


def evaluate_adapterdrop():
  return _evaluate_adapter_setting(eval_adapterdrop=True)


def main():
  run_qdlang = 'qlang' in  args.language_configs or 'dlang' in args.language_configs
  
  ablation_configs = []
  for c in args.language_configs:
    if c.lower() in ['+ra+la-inv', '+ra-la-inv']:
      ablation_configs.append(c)
  for c in ablation_configs:
    args.language_configs.pop(args.language_configs.index(c))
  run_ablation = bool(ablation_configs)
  
  run_split = 'split' in args.language_configs
  if run_split:
    args.language_configs.pop(args.language_configs.index('split'))
  
  map_results = []
  duration_results = []
  
  if run_qdlang:
    qdlang_map_results, qdlang_duration_results = evaluate_adapters()
    k = 0 if not map_results else 1 # maybe remove header
    map_results.extend(qdlang_map_results[k:])
    duration_results.extend(qdlang_duration_results[k:])
  
  if run_split:
    split_map_results, split_duration_resulst = evaluate_split_adapters()
    k = 0 if not map_results else 1 # maybe remove header
    map_results.extend(split_map_results[k:])
    duration_results.extend(split_duration_resulst[k:])
  
  if run_ablation:
    ablation_map_results, ablation_duration_results = evaluate_adapter_ablation(ablation_configs)
    k = 0 if not map_results else 1 # maybe remove header
    map_results.extend(ablation_map_results[k:])
    duration_results.extend(ablation_duration_results[k:])
  
  # evaluate_adapterdrop()
  logger.info("Done evaluating all configurations")
  print_results(duration_results, map_results)
  

if __name__ == '__main__':
  main()
