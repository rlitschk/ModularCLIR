# Parameter-Efficient Neural Reranking for Cross-Lingual and Multilingual Retrieval
Robert Litschko, Ivan Vulić, Goran Glavaš. [Parameter-Efficient Neural Reranking for Cross-Lingual and Multilingual Retrieval](https://arxiv.org/abs/2204.02292). This work builds on top of Adapters (cf. *MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer* Pfeiffer et al. 2020) and Sparse Fine-Tuning Masks (*Composable Sparse Fine-Tuning for Cross Lingual Transfer*, Ansell et al. 2021). Adapters and Masks are used to enable efficient transfer of rankers without training new models from scratch.

#### You can download our CLEF 2000-2003 query translations (Uyghur, Kyrgyz, Turkish) [here](https://madata.bib.uni-mannheim.de/401/).

## Installation
Our code has been tested with Python 3.8, we recommend to set up a new conda environment:
```
conda create --name pet-clir python=3.8
conda activate pet-clir 
pip install -r requirements.txt
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
Then install [CLEF dataloaders](https://github.com/rlitschk/clef-dataloaders) and [composable-sft](https://github.com/cambridgeltl/composable-sft) (we need to manually adjust the required Python version to 3.8):
```
git clone https://github.com/cambridgeltl/composable-sft.git
cd composable-sft
sed -i -e "s/python_requires='>=3.9'/python_requires='>=3.8'/" setup.py
pip install -e .
```
(Optional) If you want to run **NMT & BM25** (see below) on Uyghur, Kyrgyz or Turkish: 
1. Install fairseq, which is required for using the NMT model provided by [Machine Translation for Turkic Languages](https://github.com/turkic-interlingua/til-mt):
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
pip install sentencepiece sacremoses
```
2. Run `ModularCLIR/scripts/download_lowres.sh` to download the NMT model.

*Note: You may have to install a different torch/cuda environment depending on your infrastructure.*

## Training
We make example training scripts and pre-trained models available below. In order to train ranking models (monoBERT, Ranking Masks, Ranking Adapters) you need to first run `prepare_data.sh`, which downloads MS-MARCO and prepares data splits. Make sure to specify the path variables correspondingly. Scripts for training Language Adapters/Masks download Wikipedia data from [HuggingFace Datasets](https://huggingface.co/datasets/wikipedia). 

| Model                               | Training script                  | Download                                                                                                                                                    |
|-------------------------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Download and prepare MS-MARCO       | `prepare_data.sh`                | -                                                                                                                                                           |
| MonoBERT                            | `run_monoBERT_retrieval.sh`      | [Baseline (594M)](https://madata.bib.uni-mannheim.de/391/2/monobert.tar.gz)                                                                                 |
| Language Masks (LM) / Adapters (LA) | `run_{sft,adapter}_mlm.sh`       | [LM (4.5G)](https://madata.bib.uni-mannheim.de/391/6/language_masks.tar.gz), [LA (1.6G)](https://madata.bib.uni-mannheim.de/391/4/language_adapters.tar.gz) |
| Ranking Masks (RM) / Adapters (RA)  | `run_{sft,adapter}_retrieval.sh` | [RM (3.6G)](https://madata.bib.uni-mannheim.de/391/5/ranking_masks.tar.gz), [RA (256M)](https://madata.bib.uni-mannheim.de/391/3/ranking_adapters.tar.gz)   |

*Note: You can use `scripts/download.sh` to download and setup all resources at once.*

## Evaluation
To be able to run evaluation scripts below you need to set up [CLEF dataloaders](https://github.com/rlitschk/clef-dataloaders). For each model we list all other required resources, we assume they are located in `/home/usr/resources/` and the commands are run in the `PROJECT_HOME` directory. 


### Baseline: NMT & BM25
```shell
# Target location for storing (pre)ranking files and query translation files
RESOURCES_DIR=/home/usr/resources
# Target location for storing lucene index  
INDEX_HOME=$RESOURCES_DIR/index

# Transforms CLEF corpora into jsonl format and index jsonl files with pyserini
scripts/index.sh $INDEX_HOME

# Run and evaluate BM25, if necessary translate queries with EasyNMT.
python src/bm25_eval.py --save_rankings --output_dir $RESOURCES_DIR --index_dir $INDEX_HOME --lang_pairs enen dede itit fifi ruru ende enfi enit enru defi deit deru fiit firu swen soen enfa enzh
```

### Baseline: MonoBERT
Requires [Pre-ranking files](https://madata.bib.uni-mannheim.de/391/1/preranking.tar.gz) and a trained vanilla [monoBERT](https://madata.bib.uni-mannheim.de/391/2/monobert.tar.gz) model. `--mode` specifies the set of language pairs to be evaluated (**clir**: Table 1, **lowres**: Table 2, **mono**: Table 3).

```shell
# Target directory where query translations are stored
TRANSLATIONS_DIR=/home/usr/resources/translated_queries
# Directory containing a trained monoBERT model
MODEL_DIR=/home/usr/resources/monobert/checkpoint-25000
# Preranking files
PRERANKING_DIR=/home/usr/resources/preranking

python src/monobert_eval.py --model_dir $MODEL_DIR --prerank_dir $PRERANKING_DIR --mode {clir,lowres,mono} --path_query_translations $TRANSLATIONS_DIR --gpu $GPU
```

### Adapters / Sparse Fine-Tuning Masks
You can evaluate Adapters (SFTMs) with `src/adapter_eval.py` (`src/sft_eval.py`), example arguments shown below. Both require: 
- [Language Adapters](https://madata.bib.uni-mannheim.de/391/4/language_adapters.tar.gz), [Ranking Adapters](https://madata.bib.uni-mannheim.de/391/3/ranking_adapters.tar.gz) / [Language Masks](https://madata.bib.uni-mannheim.de/391/6/language_masks.tar.gz), [Ranking Masks](https://madata.bib.uni-mannheim.de/391/5/ranking_masks.tar.gz) 
- [Pre-ranking files](https://madata.bib.uni-mannheim.de/391/1/preranking.tar.gz)
- *(Optional)* `--mode lowres`: Swahili and Somali query translation files, run [NMT & BM25](https://github.com/rlitschk/ModularCLIR/#baseline-nmt--bm25) first.

Below we use the following notation for specifying language adapters (LA) and masks (LM) (`--lanuage_configs`).
- `qlang`: LA<sup>Query</sup>, LM<sup>Query</sup>
- `dlang`: LA<sup>Doc</sup>, LM<sup>Doc</sup>
- `split`/`both`: LA<sup>split</sup>, LM<sup>both</sup>

```shell
# Location of (1) trained or downloaded Adapters/SFTMs, (2) directory of preranking files and optionally (3) query translation files 
MODEL_HOME=/home/usr/resources/{adapter,sft}
PRERANKING_DIR=/home/usr/resources/preranking
TRANSLATIONS_DIR=/home/usr/resources/translated_queries
GPU=0

# Cross-lingual Evaluation args (Table 1)
--mode clir --task_rf 1 2 4 8 16 32 --language_configs dlang qlang {split,both} +ra+la-inv +ra-la-inv --model_dir $MODEL_HOME --prerank_dir $PRERANKING_DIR  --gpu $GPU  

# Low-resource languages Evaluation args (Table 2)
--mode lowres --task_rf 1 2 4 8 16 32 --language_configs dlang --path_query_translations $TRANSLATIONS_DIR --model_dir $MODEL_HOME --prerank_dir $PRERANKING_DIR  --gpu $GPU 

# Monolingual Language Transfer Evaluation args (Table 3)
--mode mono --task_rf 1 2 4 8 16 32 --language_configs dlang qlang {split,both} --model_dir $MODEL_HOME --prerank_dir $PRERANKING_DIR --gpu $GPU 
```

## Cite
If you use this repository, please consider citing our paper:
```bibtex
@inproceedings{litschko2022modularclir,
    title = "Parameter-Efficient Neural Reranking for Cross-Lingual and Multilingual Retrieval",
    author = "Litschko, Robert  and
      Vuli{\'c}, Ivan  and
      Glava{\v{s}}, Goran",
    booktitle = "Proceedings of COLING",
    year = "2022",
    pages = "1071--1082",
}
```
