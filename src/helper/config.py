import os

max_seq_len = 512
lang_rf = '2' # recommended value
preranker_mono = 'bm25' # 'fasttext'
preranker_lowres = 'fbnmt+bm25' # 'marianmt+bm25' (does not support de-ru)
preranker_cross = 'distil_mbert' # best CLIR model according to Litschko et al. 2021 (ECIR)


def get_preranker(qlang: str, dlang: str):
  if qlang == dlang:
    return preranker_mono
  if qlang == 'sw' or qlang == 'so':
    return preranker_lowres
  else:
    return preranker_cross


crosslingual_lang_pairs = [
  ("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de"),
  ("de", "fi"), ("de", "it"), ("de", "ru"),
  ("fi", "it"), ("fi", "ru")
]
monolingual_lang_pairs = [("en", "en"), ("fi", "fi"), ("de", "de"), ("it", "it"), ("ru", "ru")]
low_res_lang_pairs = [("sw", "en"), ("so", "en")]
all_lang_pairs = [
  # cross-lingual retrieval (NMT + BM25)
  ("en", "fi"), ("en", "it"), ("en", "ru"), ("en", "de"),
  ("de", "fi"), ("de", "it"), ("de", "ru"),
  ("fi", "it"), ("fi", "ru"),
  # monolingual retrieval (BM25)
  ("en", "en"), ("fi", "fi"), ("de", "de"), ("it", "it"), ("ru", "ru"),
  # low-resource query languages (NMT + BM25)
  ("so", "en"), ("sw", "en")
]


def get_language_pairs(mode: str, path_query_translations: str):
  if mode == "mono":
    return monolingual_lang_pairs
  elif mode == "clir":
    return crosslingual_lang_pairs
  else:
    assert os.path.exists(path_query_translations)
    assert len(os.listdir(path_query_translations)) > 0
    return low_res_lang_pairs
