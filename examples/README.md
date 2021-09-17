# Examples of Using DensePhrases

We provide descriptions on several examples of using DensePhrases.
For instance, based on the retrieved passages from DensePhrases, you can train a state-of-the-art open-domain question answering model called [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) by Izacard and Grave, 2021.
Or, you can build an entity linking tool with DensePhrases.
We list the examples below.

* [Basics: Multi-Granularity Text Retrieval](#basics-multi-granularity-text-retrieval)
* [Fusion-in-Decoder](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/fusion-in-decoder)
* [Entity Linking](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/entity-linking)
* [Slot Filling](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling)
* [Create Your Own Phrase Index](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/create-index)

## Basics: Multi-Granularity Text Retrieval
The most basic use of DensePhrases is to retrieve phrases, sentences, paragraphs, or documents for your query.
```python
from densephrases import DensePhrases

# Load DensePhrases (download the pre-trained model and the phrase index first)
model = DensePhrases(
    load_dir='/path/to/densephrases-multi-query-multi',
    dump_dir='/path/to/densephrases-multi_wiki-20181220/dump'
)

# Search phrases
print(model.search('Who won the Nobel Prize in peace?', retrieval_unit='phrase'))
# ['Denis Mukwege,', 'Theodore Roosevelt', 'Denis Mukwege', 'John Mott', 'Muhammad Yunus', ...]

# Search sentences
print(model.search('Why is the sky blue', retrieval_unit='sentence'))
# ['The blue color is sometimes wrongly attributed to Rayleigh scattering, which is responsible for the color of the sky.', ...]

# Search paragraphs
print(model.search('How to become a great researcher', retrieval_unit='paragraph'))
# ['... Levine said he believes the key to being a great researcher is having passion for research in and working on questions that the researcher is truly curious about. He said: "Have patience, persistence and enthusiasm and you’ll be fine."', ...]

# Search documents (Wikipedia titles)
print(model.search('What is the history of internet', retrieval_unit='document'))
# ['Computer network', 'History of the World Wide Web', 'History of the Internet', ...]
```
For batch queries, simply feed a list of queries as ``query``.
To get more detailed search results, set ``return_meta=True`` as follows:
```python
# Search phrases and get detailed results
phrases, metadata = model.search('Who won the Nobel Prize in peace?', retrieval_unit='phrase', return_meta=True)

print(phrases)
# ['Denis Mukwege,', 'Theodore Roosevelt', 'Denis Mukwege', 'John Mott', 'Muhammad Yunus', ...]

print(metadata)
# [{'context': '... The most recent as of 2018, Denis Mukwege, was awarded his Peace Prize in 2018. ...', 'title': ['List of black Nobel laureates'], 'doc_idx': 5433697, 'start_pos': 558, 'end_pos': 572, 'start_idx': 15, 'end_idx': 16, 'score': 99.670166015625, ..., 'answer': 'Denis Mukwege,'}] 
```
Note that when the model returns phrases, it also returns passages in its metadata as described in our [EMNLP paper](https://arxiv.org/abs/2109.08133).
You can also evaluate your model as follows:

```python
# Evaluate loaded DensePhrases
model.evaluate(test_path='/path/to/test_file.json')
```
