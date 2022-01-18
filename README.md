# DensePhrases

[**Getting Started**](#getting-started) | [**Lee et al., ACL 2021**](https://arxiv.org/abs/2012.12624) | [**Lee et al., EMNLP 2021**](https://arxiv.org/abs/2109.08133) | [**Demo**](http://densephrases.korea.ac.kr) | [**References**](#references) | [**License**](https://github.com/princeton-nlp/DensePhrases/blob/main/LICENSE)

<em>DensePhrases</em> is a text retrieval model that can return phrases, sentences, passages, or documents for your natural language inputs. Using billions of dense phrase vectors from the entire Wikipedia, DensePhrases searches phrase-level answers to your questions in real-time or retrieves passages for downstream tasks.

<div align="center">
  <img alt="DensePhrases Demo" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/preview-new.gif" width="750px">
</div>

Please see our ACL paper ([Learning Dense Representations of Phrases at Scale](https://arxiv.org/abs/2012.12624)) for details on how to learn dense representations of phrases and the EMNLP paper ([Phrase Retrieval Learns Passage Retrieval, Too](https://arxiv.org/abs/2109.08133)) on how to perform multi-granularity retrieval.

**\*\*\*\*\* Try out our online demo of DensePhrases [here](http://densephrases.korea.ac.kr)! \*\*\*\*\***

### Updates
* \[**Jan 18, 2022**\] [DensePhrases v1.1.0](https://github.com/princeton-nlp/DensePhrases/tree/v1.1.0) released for `transformers==4.13.0` (see [notes](https://github.com/princeton-nlp/DensePhrases/releases)).
* \[**Nov 22, 2021**\] [Test prediction files](#model-list) of `densephrases-multi-query-*` added.
* \[**Oct 10, 2021**\] See our [blog post on phrase retrieval](https://princeton-nlp.github.io/phrase-retrieval-and-beyond/) to learn more about phrase retrieval!
* \[**Sep 23, 2021**\] More examples on [entity linking](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/entity-linking), [knowledge-grounded dialouge](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/knowledge-dialogue), and [slot filling](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/slot-filling).
* \[**Sep 20, 2021**\] Pre-trained models are also available on [the Huggingface model hub](#2-pre-trained-models).
* \[**Sep 17, 2021**\] Check out updates on [multi-granularity retrieval](#getting-started), [smaller phrase indexes](#3-phrase-index) (20~60GB), and more [examples](https://github.com/princeton-nlp/DensePhrases/tree/main/examples)!
* \[**Sep 17, 2021**\] Our new [EMNLP paper](https://arxiv.org/abs/2109.08133) on phrase-based passage retrieval is out!
* \[**June 14, 2021**\] Major code updates

## Getting Started
After [installing DensePhrases](#installation) and [dowloading a phrase index](#3-phrase-index) you can easily retrieve phrases, sentences, paragraphs, or documents for your query.

https://user-images.githubusercontent.com/7017152/134703179-df5cb9d9-8151-433f-bc08-55f6fd42ed52.mp4

See [here](https://github.com/princeton-nlp/DensePhrases/tree/main/examples) for more examples such as using CPU-only mode, creating a custom index, and more.

You can also use DensePhrases to retrieve relevant documents for a dialogue or run entity linking over given texts.
```python
>>> from densephrases import DensePhrases

# Load DensePhrases for dialogue and entity linking
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-kilt-multi',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )

# Retrieve relevant documents for a dialogue
>>> model.search('I love rap music.', retrieval_unit='document', top_k=5)
['Rapping', 'Rap metal', 'Hip hop', 'Hip hop music', 'Hip hop production']

# Run entity linking for the target phrase denoted as [START_ENT] and [END_ENT]
>>> model.search('[START_ENT] Security Council [END_ENT] members expressed concern on Thursday', retrieval_unit='document', top_k=1)
['United Nations Security Council']
```
We provide more [examples](https://github.com/princeton-nlp/DensePhrases/tree/main/examples), which includes training a state-of-the-art open-domain question answering model called [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) by Izacard and Grave, 2021.

## Quick Link
* [Installation](#installation)
* [Resources: datasets, pre-trained models, phrase indexes](#resources)
* [Examples](https://github.com/princeton-nlp/DensePhrases/tree/main/examples)
* [Playing with a DensePhrases Demo](#playing-with-a-densephrases-demo)
* [Traning, Indexing and Inference](#densephrases-training-indexing-and-inference)
* [Pre-processing](#pre-processing)

## Installation
```bash
# Install torch with conda (please check your CUDA version)
conda create -n densephrases python=3.7
conda activate densephrases
conda install pytorch=1.9.0 cudatoolkit=11.0 -c pytorch

# Install apex
git clone https://www.github.com/nvidia/apex.git
cd apex
python setup.py install
cd ..

# Install DensePhrases
git clone -b v1.0.0 https://github.com/princeton-nlp/DensePhrases.git
cd DensePhrases
pip install -r requirements.txt
python setup.py develop
```

`main` branch uses `python==3.7` and `transformers==2.9.0`. See below for other versions of DensePhrases.
|              Release              | Note | Description |
|:----------------------------------:|:--------:|:--------|
| [v1.0.0](https://github.com/princeton-nlp/DensePhrases/tree/v1.0.0) | [link](https://github.com/princeton-nlp/DensePhrases/releases/tag/v1.0.0) | `transformers==2.9.0`, same as `main`| 
| [v1.1.0](https://github.com/princeton-nlp/DensePhrases/tree/v1.1.0) | [link](https://github.com/princeton-nlp/DensePhrases/releases/tag/v1.1.0) |`transformers==4.13.0` |

## Resources
Before downloading the required files below, please set the default directories as follows and ensure that you have enough storage to download and unzip the files:
```bash
# Running config.sh will set the following three environment variables:
# DATA_DIR: for datasets (including 'kilt', 'open-qa', 'single-qa', 'truecase', 'wikidump')
# SAVE_DIR: for pre-trained models or index; new models and index will also be saved here
# CACHE_DIR: for cache files from Huggingface Transformers
source config.sh
```
To download the resources described below, you can use `download.sh` as follows:
```bash
# Use bash script to download data (change data to models or index accordingly)
source download.sh
Choose a resource to download [data/wiki/models/index]: data
data will be downloaded at ...
...
Downloading data done!
```

### 1. Datasets
* [Datasets](https://nlp.cs.princeton.edu/projects/densephrases/densephrases-data.tar.gz) (1GB) - Pre-processed datasets including reading comprehension, generated questions, open-domain QA and slot filling. Download and unzip it under `$DATA_DIR` or use `download.sh`.
* [Wikipedia dumps](https://nlp.cs.princeton.edu/projects/densephrases/wikidump.tar.gz) (5GB) - Pre-processed Wikipedia dumps in different sizes. See [here](#2-creating-a-phrase-index) for more details. Download and unzip it under `$DATA_DIR` or use `download.sh`.
```bash
# Check if the download is complete
ls $DATA_DIR
kilt  open-qa  single-qa  truecase  wikidump
```

### 2. Pre-trained Models
#### Huggingface Transformers
You can use pre-trained models from the Huggingface model hub.
Any model name that starts with `princeton-nlp` (specified in `load_dir`) will be automatically translated as a model in [our Huggingface model hub](https://huggingface.co/princeton-nlp).
```python
>>> from densephrases import DensePhrases

# Load densephraes-multi-query-nq from the Huggingface model hub
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-nq',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )
```

#### Model list

|              Model                | Query-FT. | NQ | WebQ | TREC | TriviaQA | SQuAD | Description |
|:----------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [densephrases-multi](https://huggingface.co/princeton-nlp/densephrases-multi) | None | 31.9 | 25.5 | 35.7 | 44.4	| 29.3 | EM before any Query-FT. |
| [densephrases-multi-query-multi](https://huggingface.co/princeton-nlp/densephrases-multi-query-multi) | Multiple | 40.8 | 35.0 | 48.8 | 53.3 | 34.2 | Used for [demo] |

|              Model                | Query-FT. & Eval | EM | Prediction (Test) | Description |
|:----------------------------------|:--------:|:--------:|:--------:|:--------:|
| [densephrases-multi-query-nq](https://huggingface.co/princeton-nlp/densephrases-multi-query-nq) | NQ | 41.3 | [link](https://nlp.cs.princeton.edu/projects/densephrases/preds/test_preprocessed_3610_top10.pred) | - |
| [densephrases-multi-query-wq](https://huggingface.co/princeton-nlp/densephrases-multi-query-wq) | WebQ | 41.5 | [link](https://nlp.cs.princeton.edu/projects/densephrases/preds/WebQuestions-test_preprocessed_2032_top10.pred) | - |
| [densephrases-multi-query-trec](https://huggingface.co/princeton-nlp/densephrases-multi-query-trec) | TREC | 52.9 | [link](https://nlp.cs.princeton.edu/projects/densephrases/preds/CuratedTrec-test_preprocessed_694_top10.pred) | `--regex` required |
| [densephrases-multi-query-tqa](https://huggingface.co/princeton-nlp/densephrases-multi-query-tqa) | TriviaQA | 53.5 | [link](https://nlp.cs.princeton.edu/projects/densephrases/preds/test_preprocessed_11313_top10.pred) | - |
| [densephrases-multi-query-sqd](https://huggingface.co/princeton-nlp/densephrases-multi-query-sqd) | SQuAD | 34.5 | [link](https://nlp.cs.princeton.edu/projects/densephrases/preds/test_preprocessed_10570_top10.pred) | - |

**Important**: all models except `densephrases-multi` are query-side fine-tuned on the specified dataset (Query-FT.) using the phrase index [densephrases-multi_wiki-20181220](#3-phrase-index). Also note that our pre-trained models are case-sensitive models and the best results are obtained when `--truecase` is on for any lowercased queries (e.g., NQ).
* `densephrases-multi`: trained on mutiple reading comprehension datasets (NQ, WebQ, TREC, TriviaQA, SQuAD).
* `densephrases-multi-query-multi`: `densephrases-multi` query-side fine-tuned on multiple open-domain QA datasets (NQ, WebQ, TREC, TriviaQA, SQuAD).
* `densephrases-multi-query-*`: `densephrases-multi` query-side fine-tuned on each open-domain QA dataset.

For pre-trained models in other tasks (e.g., slot filling), see [examples](https://github.com/princeton-nlp/DensePhrases/tree/main/examples). Note that most pre-trained models  are the results of [query-side fine-tuning](#3-query-side-fine-tuning) `densephrases-multi`.


#### Download manually
* [Pre-trained models](https://nlp.cs.princeton.edu/projects/densephrases/outputs.tar.gz) (8GB) - All pre-trained DensePhrases models (including cross-encoder teacher models `spanbert-base-cased-*`). Download and unzip it under `$SAVE_DIR` or use `download.sh`.
```bash
# Check if the download is complete
ls $SAVE_DIR
densephrases-multi  densephrases-multi-query-nq  ...  spanbert-base-cased-squad
```

```python
>>> from densephrases import DensePhrases

# Load densephraes-multi-query-nq locally
>>> model = DensePhrases(
...     load_dir='/path/to/densephrases-multi-query-nq',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
... )
```

### 3. Phrase Index
Please note that you don't need to download this phrase index unless you want to work on the full Wikipedia scale.
* [densephrases-multi_wiki-20181220](https://nlp.cs.princeton.edu/projects/densephrases/densephrases-multi_wiki-20181220.tar.gz) (74GB) - Original phrase index (1048576_flat_OPQ96) + metadata for the entire Wikipedia (2018.12.20). Download and unzip it under `$SAVE_DIR` or use `download.sh`.

We also provide smaller phrase indexes based on more aggresive filtering (optional).
* [1048576_flat_OPQ96_medium](https://nlp.cs.princeton.edu/projects/densephrases/indexes/1048576_flat_OPQ96_medium.tar.gz) (37GB) - Medium-sized phrase index
* [1048576_flat_OPQ96_small](https://nlp.cs.princeton.edu/projects/densephrases/indexes/1048576_flat_OPQ96_small.tar.gz) (21GB) - Small-sized phrase index

These smaller indexes should be placed under `$SAVE_DIR/densephrases-multi_wiki-20181220/dump/start` along with any other indexes you downloaded.
If you only use a smaller phrase index and don't want to download the large index (74GB), you need to download [metadata](https://nlp.cs.princeton.edu/projects/densephrases/indexes/meta_compressed.pkl) (20GB) and place it under `$SAVE_DIR/densephrases-multi_wiki-20181220/dump` folder as shown below.
The structure of the files should look like:
```bash
$SAVE_DIR/densephrases-multi_wiki-20181220
└── dump
    ├── meta_compressed.pkl
    └── start
        ├── 1048576_flat_OPQ96
        ├── 1048576_flat_OPQ96_medium
        └── 1048576_flat_OPQ96_small
```
All phrase indexes are created from the same model (`densephrases-multi`) and you can use all of pre-trained models above with any of these phrase indexes.
To change the index, simply set `index_name` (or `--index_name` in `densephrases/options.py`) as follows:
```python
>>> from densephrases import DensePhrases

# Load DensePhrases with a smaller index
>>> model = DensePhrases(
...     load_dir='princeton-nlp/densephrases-multi-query-multi',
...     dump_dir='/path/to/densephrases-multi_wiki-20181220/dump',
...     index_name='start/1048576_flat_OPQ96_small'
... )
```
The performance of `densephrases-multi-query-nq` on Natural Questions (test) with different phrase indexes is shown below.

<div class="table-wrapper" markdown="block">

|              Phrase Index              | Open-Domain QA (EM) | Sentence Retrieval (Acc@1/5) | Passage Retrieval (Acc@1/5) | Size | Description |
|:----------------------------------|:---------------:|:--------:|:--------:|:--------:|:-----------------------:|
| 1048576_flat_OPQ96 | 41.3 | 48.7 / 66.4 | 52.6 / 71.5 | 60GB | evaluated with [`eval-index-psg`](https://github.com/princeton-nlp/DensePhrases/blob/main/Makefile#L477) |
| 1048576_flat_OPQ96_medium | 39.9 | 48.3 / 65.8 | 52.2 / 70.9 | 39GB | |
| 1048576_flat_OPQ96_small | 38.0 | 47.2 / 64.0 | 50.7 / 69.1 | 20GB | |

</div>

Note that the passage retrieval accuracy (Acc@1/5) is generally higher than the reported numbers in the paper since these phrase indexes return natural paragraphs instead of fixed-sized text blocks (i.e., 100 words).

## Playing with a DensePhrases Demo
You can run [the Wikipedia-scale demo](http://densephrases.korea.ac.kr) on your own server.
For your own demo, you can change the phrase index (obtained from [here](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/create-custom-index)) or the query encoder (e.g., to `densephrases-multi-query-nq`).

The resource requirement for running the full Wikipedia scale demo is:
* 50 ~ 100GB RAM (depending on the size of a phrase index)
* Single 11GB GPU (optional)

Note that you no longer need an SSD to run the demo unlike previous phrase retrieval models ([DenSPI](https://github.com/uwnlp/denspi), [DenSPI+Sparc](https://github.com/jhyuklee/sparc)). The following commands serve exactly the same demo as [here](http://densephrases.korea.ac.kr) on your `http://localhost:51997`.
```bash
# Serve a query encoder on port 1111
nohup python run_demo.py \
    --run_mode q_serve \
    --cache_dir $CACHE_DIR \
    --load_dir princeton-nlp/densephrases-multi-query-multi \
    --cuda \
    --max_query_length 32 \
    --query_port 1111 > $SAVE_DIR/logs/q-serve_1111.log &

# Serve a phrase index on port 51997 (takes several minutes)
nohup python run_demo.py \
    --run_mode p_serve \
    --index_name start/1048576_flat_OPQ96 \
    --cuda \
    --truecase \
    --dump_dir $SAVE_DIR/densephrases-multi_wiki-20181220/dump/ \
    --query_port 1111 \
    --index_port 51997 > $SAVE_DIR/logs/p-serve_51997.log &

# Below are the same but simplified commands using Makefile
make q-serve MODEL_NAME=densephrases-multi-query-multi Q_PORT=1111
make p-serve DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/ Q_PORT=1111 I_PORT=51997
```
Please change `--load_dir` or `--dump_dir` if necessary and remove `--cuda` for CPU-only version. Once you set up the demo, the log files in `$SAVE_DIR/logs/` will be automatically updated whenever a new question comes in. You can also send queries to your server using mini-batches of questions for faster inference.

```bash
# Test on NQ test set
python run_demo.py \
    --run_mode eval_request \
    --index_port 51997 \
    --test_path $DATA_DIR/open-qa/nq-open/test_preprocessed.json \
    --eval_batch_size 64 \
    --save_pred \
    --truecase

# Same command with Makefile
make eval-demo I_PORT=51997

# Result
(...)
INFO - eval_phrase_retrieval -   {'exact_match_top1': 40.83102493074792, 'f1_score_top1': 48.26451418695196}
INFO - eval_phrase_retrieval -   {'exact_match_top10': 60.11080332409972, 'f1_score_top10': 68.47386731458751}
INFO - eval_phrase_retrieval -   Saving prediction file to $SAVE_DIR/pred/test_preprocessed_3610_top10.pred
```
For more details (e.g., changing the test set), please see the targets in `Makefile` (`q-serve`, `p-serve`, `eval-demo`, etc).

## DensePhrases: Training, Indexing and Inference
In this section, we introduce a step-by-step procedure to train DensePhrases, create phrase vectors and indexes, and run inferences with the trained model.
All of our commands here are simplified as `Makefile` targets, which include exact dataset paths, hyperparameter settings, etc.

If the following test run completes without an error after the installation and the download, you are good to go!
```bash
# Test run for checking installation (takes about 10 mins; ignore the performance)
make draft MODEL_NAME=test
```

<div align="center">
  <img alt="DensePhrases Steps" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/overview_new.png" width="850px">
</div>

- A figure summarizing the overall process below

### 1. Training phrase and query encoders
To train DensePhrases from scratch, use `run-rc-nq` in `Makefile`, which trains DensePhrases on NQ (pre-processed for the reading comprehension task) and evaluate it on reading comprehension as well as on (semi) open-domain QA.
You can simply change the training set by modifying the dependencies of `run-rc-nq` (e.g., `nq-rc-data` => `sqd-rc-data` and `nq-param` => `sqd-param` for training on SQuAD).
You'll need a single 24GB GPU for training DensePhrases on reading comprehension tasks, but you can use smaller GPUs by setting `--gradient_accumulation_steps` properly.
```bash
# Train DensePhrases on NQ with Eq. 9 in Lee et al., ACL'21
make run-rc-nq MODEL_NAME=densephrases-nq
```

`run-rc-nq` is composed of the six commands as follows (in case of training on NQ):
1. `make train-rc ...`: Train DensePhrases on NQ with Eq. 9 (L = lambda1 L\_single + lambda2 L\_distill + lambda3 L\_neg) with generated questions.
2. `make train-rc ...`: Load trained DensePhrases in the previous step and further train it with Eq. 9 with pre-batch negatives.
3. `make gen-vecs`: Generate phrase vectors for D\_small (= set of all passages in NQ dev).
4. `make index-vecs`: Build a phrase index for D\_small.
5. `make compress-meta`: Compresss metadata for faster inference.
6. `make eval-index ...`: Evaluate the phrase index on the development set questions.

At the end of step 2, you will see the performance on the reading comprehension task where a gold passage is given (about 72.0 EM on NQ dev). Step 6 gives the performance on the semi-open-domain setting (denoted as D\_small; see Table 6 in the paper) where the entire passages from the NQ development set is used for the indexing (about 62.0 EM with NQ dev questions). The trained model will be saved under `$SAVE_DIR/$MODEL_NAME`. Note that during the single-passage training on NQ, we exclude some questions in the development set, whose annotated answers are found from a list or a table.

###  2. Creating a phrase index
Let's assume that you have a pre-trained DensePhrases named `densephrases-multi`, which can also be downloaded from [here](#2-pre-trained-models).
Now, you can generate phrase vectors for a large-scale corpus like Wikipedia using `gen-vecs-parallel`.
Note that you can just download [the phrase index for the full Wikipedia scale](#3-phrase-index) and skip this section.
```bash
# Generate phrase vectors in parallel for a large-scale corpus (default = wiki-dev)
make gen-vecs-parallel MODEL_NAME=densephrases-multi START=0 END=8
```
The default text corpus for creating phrase vectors is `wiki-dev` located in `$DATA_DIR/wikidump`. We have three options for larger text corpora:
- `wiki-dev`: 1/100 Wikipedia scale (sampled), 8 files
- `wiki-dev-noise`: 1/10 Wikipedia scale (sampled), 500 files
- `wiki-20181220`: full Wikipedia (20181220) scale, 5621 files

The `wiki-dev*` corpora also contain passages from the NQ development set, so that you can track the performance of your model with an increasing size of the text corpus (usually decreases as it gets larger). The phrase vectors will be saved as hdf5 files in `$SAVE_DIR/$(MODEL_NAME)_(data_name)/dump` (e.g., `$SAVE_DIR/densephrases-multi_wiki-dev/dump`), which will be referred to `$DUMP_DIR` below.

#### Parallelization
`START` and `END` specify the file index in the corpus (e.g., `START=0 END=8` for `wiki-dev` and `START=0 END=5621` for `wiki-20181220`).  Each run of `gen-vecs-parallel` only consumes 2GB in a single GPU, and you can distribute the processes with different `START` and `END` using slurm or shell script (e.g., `START=0 END=200`, `START=200 END=400`, ..., `START=5400 END=5621`). Distributing 28 processes on 4 24GB GPUs (each processing about 200 files) can create phrase vectors for `wiki-20181220` in 8 hours. Processing the entire Wikiepdia requires up to 500GB and we recommend using an SSD to store them if possible (a smaller corpus can be stored in a HDD).

After generating the phrase vectors, you need to create a phrase index for the sublinear time search of phrases. Here, we use IVFOPQ for the phrase index.
```bash
# Create IVFOPQ index for a set of phrase vectors
make index-vecs DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump/
```

For `wiki-dev-noise` and `wiki-20181220`, you need to modify the number of clusters to 101,372 and 1,048,576, respectively (simply change `medium1-index` in `ìndex-vecs` to `medium2-index` or `large-index`). For `wiki-20181220` (full Wikipedia), this takes about 1~2 days depending on the specification of your machine and requires about 100GB RAM. For IVFSQ as described in the paper, you can use `index-add` and `index-merge` to distribute the addition of phrase vectors to the index.

You also need to compress the metadata (saved in hdf5 files together with phrase vectors) for a faster inference of DensePhrases. This is mandatory for the IVFOPQ index.
```bash
# Compress metadata of wiki-dev
make compress-meta DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump
```

For evaluating the performance of DensePhrases with your phrase indexes, use `eval-index`.
```bash
# Evaluate on the NQ test set questions
make eval-index MODEL_NAME=densephrases-multi DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump/
```

### 3. Query-side fine-tuning
Query-side fine-tuning makes DensePhrases a versatile tool for retrieving multi-granularity text for different types of input queries. While query-side fine-tuning can also improve the performance on QA datasets, it can be used to adapt DensePhrases to **non-QA style input queries** such as "subject [SEP] relation" to retrieve object entities or "I love rap music." to retrieve relevant documents on rapping.

First, you need a phrase index for the full Wikipedia (`wiki-20181220`), which can be simply downloaded [here](#3-phrase-index), or a custom phrase index as described [here](https://github.com/princeton-nlp/DensePhrases/tree/main/examples/create-custom-index).
Given your query-answer or query-document pairs pre-processed as json files in `$DATA_DIR/open-qa` or `$DATA_DIR/kilt`, you can easily query-side fine-tune your model. For instance, the training set of T-REx (`$DATA_DIR/kilt/trex/trex-train-kilt_open_10000.json`) looks as follows:
```
{
    "data": [
        {
            "id": "111ed80f-0a68-4541-8652-cb414af315c5",
            "question": "Effie Germon [SEP] occupation",
            "answers": [
                "actors",
                ...
            ]
        },
        ...
    ]
}
```
The following command query-side fine-tunes `densephrases-multi` on T-REx.
```bash
# Query-side fine-tune on T-REx (model will be saved as MODEL_NAME)
make train-query MODEL_NAME=densephrases-multi-query-trex DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/
```
Note that the pre-trained query encoder is specified in `train-query` as `--load_dir $(SAVE_DIR)/densephrases-multi` and a new model will be saved as `densephrases-multi-query-trex` as specified in `MODEL_NAME`. You can also train on different datasets by changing the dependency `trex-open-data` to `*-open-data` (e.g., `ay2-kilt-data` for entity linking).

### 4. Inference
With any DensePhrases query encoders (e.g., `densephrases-multi-query-nq`) and a phrase index (e.g., `densephrases-multi_wiki-20181220`), you can test your queries as follows and the retrieval results will be saved as a json file with the `--save_pred` option:

```bash
# Evaluate on Natural Questions
make eval-index MODEL_NAME=densephrases-multi-query-nq DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/

# If the demo is being served on http://localhost:51997
make eval-demo I_PORT=51997
```
For the evaluation on different datasets, simply change the dependency of `eval-index` (or `eval-demo`) accordingly (e.g., `nq-open-data` to `trec-open-data` for the evaluation on CuratedTREC).

## Pre-processing
At the bottom of `Makefile`, we list commands that we used for pre-processing the datasets and Wikipedia. For training question generation models (T5-large), we used [https://github.com/patil-suraj/question\_generation](https://github.com/patil-suraj/question_generation) (see also [here](https://github.com/princeton-nlp/DensePhrases/blob/main/scripts/question_generation/generate_squad.py) for QG). Note that all datasets are already pre-processed including the generated questions, so you do not need to run most of these scripts. For creating test sets for custom (open-domain) questions, see `preprocess-openqa` in `Makefile`.

## Questions?
Feel free to email Jinhyuk Lee `(jinhyuklee@princeton.edu)` for any questions related to the code or the paper. You can also open a Github issue. Please try to specify the details so we can better understand and help you solve the problem.

## References
Please cite our paper if you use DensePhrases in your work:
```bibtex
@inproceedings{lee2021learning,
    title={Learning Dense Representations of Phrases at Scale},
    author={Lee, Jinhyuk and Sung, Mujeen and Kang, Jaewoo and Chen, Danqi},
    booktitle={Association for Computational Linguistics (ACL)},
    year={2021}
}
```
```bibtex
@inproceedings{lee2021phrase,
    title={Phrase Retrieval Learns Passage Retrieval, Too},
    author={Lee, Jinhyuk and Wettig, Alexander and Chen, Danqi},
    booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year={2021},
}
```

## License
Please see LICENSE for details.

[demo]: http://densephrases.korea.ac.kr
