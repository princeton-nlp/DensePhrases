# Slot Filling

## Pre-trained Models
|              Model              | Query-FT. & Eval | KILT-Accuracy | Description |
|:-------------------------------|:--------:|:--------:|:--------:|
| [densephrases-multi-query-trex](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-trex.tar.gz) | T-REx | 22.3 | Result from [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview) |
| [densephrases-multi-query-zsre](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-zsre.tar.gz) | Zero-shot RE | 40.0 | |

## How to Use
```python
>>> from densephrases import DensePhrases

# Load densephraes-multi-query-trex locally
>>> model = DensePhrases(
...     load_dir='/path/to/densephrases-multi-query-trex',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )

# Slot filling queries are in the format of 'Subject [SEP] Relation'
>>> model.search('Superman [SEP] father', retrieval_unit='phrase', top_k=5)
['Jor-El', 'Clark Kent', 'Jor-El', 'Jor-El', 'Jor-El']

>>> model.search('Cirith Ungol [SEP] genre', retrieval_unit='phrase', top_k=5)
['heavy metal', 'doom metal', 'metal', 'Elvish', 'madrigal comedy']
```

### Evaluation
```python
>>> import os

# Evaluate loaded DensePhrases on T-REx (KILT)
>>> model.evaluate(
...     test_path=os.path.join(os.environ['DATA_DIR'], 'kilt/trex/trex-dev-kilt_open.json'),
...     is_kilt=True, title2wikiid_path=os.path.join(os.environ['DATA_DIR'], 'wikidump/title2wikiid.json'),
...     kilt_gold_path=os.path.join(os.environ['DATA_DIR'], 'kilt/trex/trex-dev-kilt.jsonl'), agg_strat='opt2',
... )
```

For test accuracy, use `trex-test-kilt_open.json` instead and submit the prediction file (saved as `$SAVE_DIR/densephrases-multi-query-trex/pred-kilt/densephrases-multi-query-trex_trex-test-kilt_open_5000.jsonl`) to [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview).
For zero-shot relation extraction, follow the same process with files specified in the `zsre-kilt-data` target in [Makefile](https://github.com/princeton-nlp/DensePhrases/blob/main/Makefile).
You can also evaluate the model with Makefile `eval-index` target by simply chaning the dependency to `trex-kilt-data` or `zsre-kilt-data`.
