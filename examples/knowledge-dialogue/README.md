# Knowledge-Grounded Dialogue

## Pre-trained Models
|              Model              | Query-FT. & Eval | R-Precision| Description |
|:-------------------------------|:--------:|:--------:|:--------:|
| [densephrases-multi-query-wow](https://huggingface.co/princeton-nlp/densephrases-multi-query-wow) | Wizard of Wikipedia (WoW) | 47.0 | Result from [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview) |
| [densephrases-multi-query-kilt-multi](https://huggingface.co/princeton-nlp/densephrases-multi-query-kilt-multi) | Multiple / WoW | 55.7 | Trained on multiple KILT tasks |

## How to Use
```python
>>> from densephrases import DensePhrases

# Load densephraes-multi-query-wow
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-wow',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )

# Feed a dialogue as a query
>>> model.search('I love rap music.', retrieval_unit='document', top_k=10)
['Rapping', 'Hip hop', 'Rap metal', 'Hip hop music', 'Rapso', 'Battle rap', 'Rape', 'Eurodance', 'Chopper (rap)', 'Rape culture']

>>> model.search('Have you heard of Yamaha? They started as a piano manufacturer in 1887!', retrieval_unit='document', top_k=5)
['Yamaha Corporation', 'Yamaha Drums', 'Tōkai Gakki', 'Suzuki Musical Instrument Corporation', 'Supermoto']

# You can get more metadata on the document by setting return_meta=True
>>> doc, meta = model.search('I love rap music.', retrieval_unit='document', top_k=1, return_meta=True)
>>> meta
[{'context': 'Rap is usually delivered over a beat, typically provided by a DJ, turntablist, ...', 'title': ['Rapping'], 'doc_idx': 4096192, 'start_pos': 647, 'end_pos': 660, 'start_idx': 91, 'end_idx': 93, 'score': 53.58412170410156, ... 'answer': 'hip-hop music'}]
```

### Evaluation
```python
>>> import os

# Evaluate loaded DensePhrases on Wizard of Wikipedia
>>> model.evaluate(
...     test_path=os.path.join(os.environ['DATA_DIR'], 'kilt/wow/wow-dev-kilt_open.json'),
...     is_kilt=True, title2wikiid_path=os.path.join(os.environ['DATA_DIR'], 'wikidump/title2wikiid.json'),
...     kilt_gold_path=os.path.join(os.environ['DATA_DIR'], 'kilt/wow/wow-dev-kilt.jsonl'), agg_strat='opt2', max_query_length=384
... )
```

For test accuracy, use `wow-test-kilt_open.json` instead and submit the prediction file (saved as `$SAVE_DIR/densephrases-multi-query-wow/pred-kilt/*.jsonl`) to [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview).
You can also evaluate the model with Makefile `eval-index` target by simply chaning the dependency.
