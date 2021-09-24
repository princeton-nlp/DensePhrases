# Entity Linking

## Pre-trained Models
|              Model              | Query-FT. & Eval | R-Precision| Description |
|:-------------------------------|:--------:|:--------:|:--------:|
| [densephrases-multi-query-ay2](https://huggingface.co/princeton-nlp/densephrases-multi-query-ay2) | AIDA CoNLL-YAGO (AY2) | 61.6 | Result from [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview) |
| [densephrases-multi-query-kilt-multi](https://huggingface.co/princeton-nlp/densephrases-multi-query-kilt-multi) | Multiple / AY2 | 68.4 | Trained on multiple KILT tasks |

## How to Use
```python
>>> from densephrases import DensePhrases

# Load densephraes-multi-query-ay2
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-ay2',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )

# Entities need to be surrounded by [START_ENT] and [END_ENT] tags
>>> model.search('West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat [START_ENT] Somerset [END_ENT] by an innings and 39 runs', retrieval_unit='document', top_k=1)
['Somerset County Cricket Club']

>>> model.search('[START_ENT] Security Council [END_ENT] members expressed concern on Thursday', retrieval_unit='document', top_k=1)
['United Nations Security Council']
```

### Evaluation
```python
>>> import os

# Evaluate loaded DensePhrases on AIDA CoNLL-YAGO (KILT)
>>> model.evaluate(
...     test_path=os.path.join(os.environ['DATA_DIR'], 'kilt/ay2/aidayago2-dev-kilt_open.json'),
...     is_kilt=True, title2wikiid_path=os.path.join(os.environ['DATA_DIR'], 'wikidump/title2wikiid.json'),
...     kilt_gold_path=os.path.join(os.environ['DATA_DIR'], 'kilt/ay2/aidayago2-dev-kilt.jsonl'), agg_strat='opt2', max_query_length=384
... )
```

For test accuracy, use `aidayago2-test-kilt_open.json` instead and submit the prediction file (saved as `$SAVE_DIR/densephrases-multi-query-ay2/pred-kilt/*.jsonl`) to [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview).
For WNED-WIKI and WNED-CWEB, follow the same process with files specified in the `wned-kilt-data` and `cweb-kilt-data` targets in [Makefile](https://github.com/princeton-nlp/DensePhrases/blob/main/Makefile).
You can also evaluate the model with Makefile `eval-index` target by simply chaning the dependency.
