# DensePhrases Examples

We provide descriptions on how to use DensePhrases for different applications.
For instance, based on the retrieved passages from DensePhrases, you can train a state-of-the-art open-domain question answering model called [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) by Izacard and Grave, 2021, or you can run entity linking with DensePhrases.

* [Basics: Multi-Granularity Text Retrieval](#basics-multi-granularity-text-retrieval)
* [Create a Custom Phrase Index](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/create-custom-index)
* [Open-Domain QA with Fusion-in-Decoder](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/fusion-in-decoder)
* [Entity Linking](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/entity-linking)
* [Knowledge-grounded Dialogue](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/knowledge-dialogue)
* [Slot Filling](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling)

## Basics: Multi-Granularity Text Retrieval
The most basic use of DensePhrases is to retrieve phrases, sentences, paragraphs, or documents for your query.
```python
>>> from densephrases import DensePhrases

# Load DensePhrases
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-multi',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump'
... )

# Search phrases
>>> model.search('Who won the Nobel Prize in peace?', retrieval_unit='phrase', top_k=5)
['Denis Mukwege,', 'Theodore Roosevelt', 'Denis Mukwege', 'John Mott', 'Mother Teresa']

# Search sentences
>>> model.search('Why is the sky blue', retrieval_unit='sentence', top_k=1)
['The blue color is sometimes wrongly attributed to Rayleigh scattering, which is responsible for the color of the sky.']

# Search paragraphs
>>> model.search('How to become a great researcher', retrieval_unit='paragraph', top_k=1)
['... Levine said he believes the key to being a great researcher is having passion for research in and working on questions that the researcher is truly curious about. He said: "Have patience, persistence and enthusiasm and you’ll be fine."']

# Search documents (Wikipedia titles)
>>> model.search('What is the history of internet', retrieval_unit='document', top_k=3)
['Computer network', 'History of the World Wide Web', 'History of the Internet']
```

For batch queries, simply feed a list of queries as ``query``.
To get more detailed search results, set ``return_meta=True`` as follows:
```python
# Search phrases and get detailed results
>>> phrases, metadata = model.search(['Who won the Nobel Prize in peace?', 'Name products of Apple.'], retrieval_unit='phrase', return_meta=True)

>>> phrases[0]
['Denis Mukwege,', 'Theodore Roosevelt', 'Denis Mukwege', 'John Mott', 'Muhammad Yunus', ...]

>>> metadata[0]
[{'context': '... The most recent as of 2018, Denis Mukwege, was awarded his Peace Prize in 2018. ...', 'title': ['List of black Nobel laureates'], 'doc_idx': 5433697, 'start_pos': 558, 'end_pos': 572, 'start_idx': 15, 'end_idx': 16, 'score': 99.670166015625, ..., 'answer': 'Denis Mukwege,'}, ...] 
```
Note that when the model returns phrases, it also returns passages in its metadata as described in our [EMNLP paper](https://arxiv.org/abs/2109.08133).<br>

### CPU-only Mode
```python
# Load DensePhrases in CPU-only mode
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-multi',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
...     device='cpu',
...     max_query_length=24, # reduce the maximum query length for a faster query encoding (optional)
... )
```

### Changing the Index or the Encoder
```python
# Load DensePhrases with a smaller phrase index
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-multi',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
...     index_name='start/1048576_flat_OPQ96_small'
... )

# Change the DensePhrases encoder to 'princeton-nlp/densephrases-multi-query-tqa' (trained on TriviaQA)
>>> model.set_encoder('princeton-nlp/densephrases-multi-query-tqa')
```

### Evaluation
```python
>>> import os

# Evaluate loaded DensePhrases on Natural Questions
>>> model.evaluate(test_path=os.path.join(os.environ['DATA_DIR'], 'open-qa/nq-open/test_preprocessed.json'))
```
