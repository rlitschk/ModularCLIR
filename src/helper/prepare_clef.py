"""
This script reads CLEF corpora and creates for each corpus a jsonl file, each line consisting of document id and 
document content. This is required input to run `PROJECT_HOME/scripts/index.sh` and `PROJECT_HOME/src/bm25_eval.py`. 
"""

import codecs
import os
import json
import argparse

from clef_dataloader import load_documents


def serialize_jsonl_files(year, tgt_dir):
  for lang in ["en", "de", "it", "fi", "ru"]:
    lang_dir = os.path.join(tgt_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)
    print(f"indexing {lang} ({lang_dir})")
    doc_ids, documents = load_documents(language=lang, year=year)
    print("documents loaded")
    jsonl = [json.dumps({"id": _id, "contents": doc}, ensure_ascii=False) + "\n" for _id, doc in zip(doc_ids, documents)]
    tgt_file = os.path.join(lang_dir, f"corpus_CLEF{year}_{lang}.jsonl")
    if lang != "ru":
      with open(tgt_file, "w") as f:
        f.writelines(jsonl)
    else:
      # encoding = 'UTF-8' if "russian" == lang else 'ISO-8859-1'
      with codecs.open(tgt_file, encoding='UTF-8', mode='w') as f:
        f.writelines(jsonl)
    print(f"done")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--year", default="2003", type=str)
  parser.add_argument("--tgt_dir", type=str)
  args = parser.parse_args()
  serialize_jsonl_files(year=args.year, tgt_dir=args.tgt_dir)
