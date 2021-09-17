# Examples of Using DensePhrases

We provide descriptions on several examples of using DensePhrases.
For instance, based on the retrieved passages from DensePhrases, you can train a state-of-the-art open-domain question answering model called [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) by Izacard and Grave, 2021.
Or, you can build an entity linking tool with DensePhrases.
We list the examples below.

* [Basics: Multi-Granularity Text Retrieval](#basics:-multi-granularity-text-retrieval)
* [Fusion-in-Decoder](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/fusion-in-decoder)
* [Entity Linking](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling)
* [Slot Filling](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling)
* [Create Your Own Index](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling)

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

You can also evaluate your model as follows:

```python
# Evaluate loaded DensePhrases
model.evaluate(test_path='/path/to/test_file.json')
```
