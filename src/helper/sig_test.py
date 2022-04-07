from scipy.stats import ttest_ind
import numpy as np
import os
from collections import defaultdict

basedir = "/home/usr/resources/"
vanilla = f"{basedir}monobert/checkpoint-25000/precision_values"
adapter = f"{basedir}adapter/ir/rf_2_16/checkpoint-625000/precision_values"
adpter_no_inv = f"{basedir}adapter/ir_-INV/rf_2_16/checkpoint-625000/precision_values"
sfts = f"{basedir}sft/ir_625k/rf_2_2/precision_values"

xling_lang_pairs = [
  ("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de"),
  ("de", "fi"), ("de", "it"), ("de", "ru"),
  ("fi", "it"), ("fi", "ru")
]

vanilla_values = []
model2precision_values = defaultdict(list)
models = [
  ("vanilla", vanilla), 
  # ("adapter", adpter),
  ("adapter-qlang", os.path.join(adapter, f"#####_setting=q-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt")),
  ("adapter-dlang", os.path.join(adapter, f"#####_setting=d-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt")),
  ("adapter-split", f"{basedir}sft/ir_625/rf_2_16/precision_values/#####_setting=split-adapter_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt"),
  ("adapter-LANG-INV", f"{basedir}sft/ir_625/rf_2_16/precision_values/#####_-LANG-INV_setting=d-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt"),
  ("adapter+LANG-INV", f"{basedir}sft/ir_625/rf_2_16/precision_values/#####_+LANG-INV_setting=d-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt"),
  ("sft-qlang", os.path.join(sfts, f"#####_setting=q-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt")),
  ("sft-dlang", os.path.join(sfts, f"#####_setting=d-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt")),
  ("sft-qdlang", os.path.join(sfts, f"#####_setting=qd-lang_preranker=distilbert-multilingual-nli-stsb-quora-ranking/precision_values.txt"))
]

for qlang, dlang in xling_lang_pairs:
  if qlang == dlang:
    preranker = "bm25"
  else:
    preranker = "distilbert-multilingual-nli-stsb-quora-ranking"

  for model, filepath in models:  
    if model == "vanilla":
      file = os.path.join(filepath, f"{qlang}-{dlang}_preranker={preranker}/precision_values.txt")
    else:
      file = filepath.replace("#####", f"{qlang}-{dlang}")
      
    with open(file, "r") as f:
      lines = [float(l.strip()) for l in f.readlines()]
    model2precision_values[model].extend(lines)
  # '/home/usr/resources/sft/ir_625/rf_2_16/precision_values/en-fi_setting=split-adapter_preranker=distilbert-multilingual-nli-stsb-quora-ranking'
  # /home/usr/resources/sft/ir_625/rf_2_16/precision_values/en-it_setting=split-adapter_preranker=distilbert-multilingual-nli-stsb-quora-ranking
  print()

print()
for model, _ in models:
  pvalue = ttest_ind(model2precision_values[model], model2precision_values["vanilla"])[1]
  a_mean = np.mean(model2precision_values[model]).round(3)
  b_mean = np.mean(model2precision_values["vanilla"]).round(3)
  print(f"{model}\t{a_mean}\t{b_mean}\t{pvalue}")