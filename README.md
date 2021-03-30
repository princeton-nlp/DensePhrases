# DensePhrases

<div align="center">
  <img alt="DensePhrases Demo" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/preview.gif" width="750px">
</div>

<em>DensePhrases</em> provides answers to your natural language questions from the entire Wikipedia in real-time. While it efficiently searches the answers out of 60 billion phrases in Wikipedia, it is also very accurate having competitive accuracy with state-of-the-art open-domain QA models.  Please see our paper [Dense Representations of Phrases at Scale (Lee et al., 2020)](https://arxiv.org/abs/2012.12624) for more details.

**\*\*\*\*\* You can try out our online demo of DensePhrases [here](http://densephrases.korea.ac.kr)! \*\*\*\*\***

## Quick Links
* [Installation](#installation)
* [Playing with a DensePhrases Demo](#playing-with-a-densephrases-demo)
* [Traning, Indexing and Inference](#densephrases-training-indexing-and-inference)
* [Pre-processing](#pre-processing)

## Installation
```bash
# Use conda & pip
conda create -n dph python=3.7
conda activate dph
conda install pytorch cudatoolkit=11.0 -c pytorch
pip install faiss-gpu==1.6.5 h5py tqdm transformers==2.9.0 blosc ujson rouge wandb nltk flask flask_cors tornado requests-futures

# Install apex
git clone https://www.github.com/nvidia/apex
cd apex
python setup.py install
```
Please check your CUDA version before the installation and modify it accordingly. Since it can be tricky to install a recent version of PyTorch together with the GPU version of Faiss, please post Github issues if you have any problem (See [Training DensePhrases](#densephrases-training-indexing-and-inference) to check whether the installation is complete).

Before downloading the required files below, please set the default directories as follows and ensure that you have enough storage to download and unzip the files:
```bash
# Running config.sh will set the following three environment variables:
# DPH_DATA_DIR: for datasets (including 'kilt', 'open-qa', 'single-qa', 'truecase', 'wikidump')
# DPH_SAVE_DIR: for pre-trained models or dumps; new models and dumps will also be saved here
# DPH_CACHE_DIR: for cache files from huggingface transformers
source config.sh
```

### 1. Datasets
* [Datasets](https://nlp.cs.princeton.edu/projects/densephrases/dph-data.tar.gz) (6GB) - All pre-processed datasets used in our experiments including reading comprehension, open-domain QA, slot filling, pre-processed Wikipedia. Download and unzip it under `DPH_DATA_DIR`.
```bash
ls $DPH_DATA_DIR
kilt  open-qa  single-qa  truecase  wikidump
```

### 2. Pre-trained Models
* [Pre-trained models](https://nlp.cs.princeton.edu/projects/densephrases/outputs.tar.gz) (5GB) - Pre-trained DensePhrases and cross-encoder teacher models. Download and unzip it under `DPH_SAVE_DIR`.
```bash
ls $DPH_SAVE_DIR
dph-nqsqd-pb2  dph-nqsqd-pb2_pq96-multi6  dph-nqsqd-pb2_pq96-nq-10  spanbert-base-cased-nq  spanbert-base-cased-sqdnq  spanbert-base-cased-squad
```
- `dph-nqsqd-pb2`                      : DensePhrases (C\_phrase = {NQ, SQuAD}) before any query-side fine-tuning
- `dph-nqsqd-pb2_pq96-nq-10`          : DensePhrases query-side fine-tuned on NQ (PQ index, NQ=40.9 EM)
- `dph-nqsqd-pb2_pq96-multi6`         : DensePhrases query-side fine-tuned on 4 open-domain QA (NQ, WQ, TREC, TQA) + 2 slot filling datasets (PQ index, NQ=40.3 EM); Used for the [demo]
- `spanbert-base-cased-*`             : cross-encoder teacher models trained on \*

### 3. Phrase Index
* [DensePhrases-IVFPQ96](https://nlp.cs.princeton.edu/projects/densephrases/dph-nqsqd-pb2_20181220_concat.tar.gz) (88GB) - Phrase index for the 20181220 version of Wikipedia. Download and unzip it under `DPH_SAVE_DIR`.
```bash
ls $DPH_SAVE_DIR
...  dph-nqsqd-pb2_20181220_concat
```
Since hosting the 320GB phrase index (+500GB original vectors for query-side fine-tuning) - the phrase index described in our paper - is costly, we provide an index with a much smaller size, which includes our recent efforts to reduce the size of the phrase index with [Product Quantization](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf) (IVFPQ). With IVFPQ, you do not need any SSDs for the real-time inference (the index is loaded on RAM), and you can also reconstruct the phrase vectors from it for the query-side fine-tuning (hence do not need the additional 500GB).

For the reimplementation of DensePhrases with IVFSQ4 as described in the paper, see [Training DensePhrases](#densephrases-training-indexing-and-inference).

## Playing with a DensePhrases Demo
There are two ways of using DensePhrases.
1. You can simply use the [demo] that we are serving on our server. The running demo is using `dph-nqsqd-pb2_pq96-multi6` (NQ=40.3 EM) as a query encoder and `dph-nqsqd-pb2_20181220_concat` as a phrase index.
2. You can install the demo on your own server, which enables you to change the query encoder (e.g., to `dph-nqsqd-pb2_pq96-nq-10`) or to process multiple queries in parallel (using HTTP POST). We recommend installing your own demo as described below since our demo can be unstable due to a large number of requests. Also, [query-side fine-tuning](#3-query-side-fine-tuning) is only available to those who installed DensePhrases on their server.

The minimum resource requirement for running the demo is:
* Single 11GB GPU
* 125GB RAM
* 100GB HDD

Note that you no longer need any SSDs to run the demo unlike previous phrase retrieval models ([DenSPI](https://github.com/uwnlp/denspi), [DenSPI+Sparc](https://github.com/jhyuklee/sparc)), but setting `$DPH_SAVE_DIR` to an SSD can reduce the loading time of the demo. The following commands serve exactly the same demo as [here](http://densephrases.korea.ac.kr) on your `http://localhost:51997`.
```bash
# Serve a query encoder on port 1111
make q-serve MODEL_NAME=dph-nqsqd-pb2_pq96-multi6 Q_PORT=1111

# Serve a phrase index on port 51997 (takes several minutes)
make p-serve DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_20181220_concat/dump/ Q_PORT=1111 I_PORT=51997
```
You can change the query encoder or the phrase index accordingly. Once you set up the demo, the log files in `$DPH_SAVE_DIR/logs/` will be automatically updated whenever a new question comes in. You can also send queries to your server using mini-batches of questions for faster inference.

```bash
# Test on NQ test set (takes 60~90 sec)
make eval-od-req I_PORT=51997
(...)
INFO - densephrases.experiments.run_open -   {'exact_match_top1': 40.30470914127424, 'f1_score_top1': 47.18394271164363}
INFO - densephrases.experiments.run_open -   {'exact_match_top10': 63.57340720221607, 'f1_score_top10': 72.15437717099778}
INFO - densephrases.experiments.run_open -   Saving prediction file to $DPH_SAVE_DIR/pred/test_preprocessed_3610.pred
```
For more details (e.g., changing the test set), please see the targets in `Makefile` (`q-serve`, `p-serve`, `eval-od-req`, etc).

## DensePhrases: Training, Indexing and Inference
In this section, we introduce the steps to train DensePhrases from scratch, create phrase dumps and indexes, and running inferences with the trained model (which can be also used as a demo described above). The minimum requirement is as follows:
- Single 24GB GPU (for training)
- up to 150GB RAM (for creating a phrase index of the entire Wikipedia)
- up to 500GB storage (for creating a phrase dump of the entire Wikipedia)

All of our commands below are specified as `Makefile` targets, which include dataset paths, hyperparameter settings, etc.
Before training DensePhrases, run the following command to check whether the installation is complete. If this command runs without an error, you are good to go!
```bash
# Test run for checking installation (ignore the performance)
make draft MODEL_NAME=test
```

<div align="center">
  <img alt="DensePhrases Steps" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/steps.png" width="850px">
</div>

- A figure summarizing the overall process below

### 1. Training phrase and query encoders
To train DensePhrase from scratch, use `train-single-nq`, which trains DensePhrases on NQ (pre-processed for the reading comprehension setting). You can simply change the training set by modifying the dependencies of `train-single-nq` (e.g., `nq-single-data` => `sqd-single-data` and `nq-param` => `sqd-param` for training on SQuAD).
```bash
# Train DensePhrases on NQ with Eq. 9
make train-single-nq MODEL_NAME=dph-nq
```

`train-single-nq` is composed of the four consecutive commands as follows (in case of training on NQ):
1. `make train-single ...`: Train DensePhrases on NQ with Eq. 9 (L = lambda1 L\_single + lambda2 L\_distill + lambda3 L\_neg) with in-batch negatives and generated questions.
2. `make train-single ...`: Load trained DensePhrases in the previous step and further train it with Eq. 9 with pre-batch negatives (dump D\_small at the end).
3. `make index-sod`: Create a phrase index for D\_small
4. `make eval-sod ...`: Evaluate the development set with D\_small

At the end of step 2, you will see the performance on the reading comprehension setting where a gold passage is given (72.0 EM on NQ dev). Step 4 gives the performance on the semi-open-domain setting (denoted as D\_small; see Table 6 in the paper.) where the entire passages from the NQ development set is used for the indexing (64.0 EM with NQ dev questions). The trained model will be saved under `$DPH_SAVE_DIR/$MODEL_NAME`. Note that during the single-passage training on NQ, we exclude some questions in the development set, whose annotated answers are found from a list or a table.

###  2. Creating a phrase index
Now let's assume that you have a model trained on NQ + SQuAD named `dph-nqsqd-pb2`, which can also be downloaded from [here](#2-pre-trained-models).
You can make a bigger phrase dump using `dump-large` as follows:
```bash
# Create large-scale phrase dumps with a trained model (default = dev_wiki)
make dump-large MODEL_NAME=dph-nqsqd-pb2 START=0 END=8
```
The default text corpus for creating phrase dump is `dev_wiki` located in `$DPH_DATA_DIR/wikidump`. We have three options for larger text corpora:
- `dev_wiki`: 1/100 Wikipedia scale (sampled), 8 files
- `dev_wiki_noise`: 1/10 Wikipedia scale (sampled), 500 files
- `20181220_concat`: full Wikipedia (20181220) scale, 5621 files

The `dev_wiki*` corpora contain passages from the NQ development set, so that you can track the performance of your model depending on the size of the text corpus (usually decreases as it gets larger). The phrase dump will be saved as hdf5 files in `$DPH_SAVE_DIR/$(MODEL_NAME)_(data_name)/dump` (`$DPH_SAVE_DIR/dph-nqsqd-pb2_dev_wiki/dump` in this case), which will be referred to `$DUMP_DIR`.

#### Parallelization
`START` and `END` specify the file index in the corpus (e.g., `START=0 END=8` for `dev_wiki` and `START=0 END=5621` for `20181220_concat`).  Each run of `dump-large` only consumes 2GB of a single GPU, and you can distribute the processes with different `START` and `END` (use slurm or shell scripts). Distributing 28 processes on 4 24GB GPUs (each processing 200 files) can create a phrase dump for `20181220_concat` in 8 hours.

After creating the phrase dump, you need to create a phrase index (or a MIPS index) for the sublinear time search of phrases. In our paper, we used IVFSQ4 for the phrase index.
```bash
# Create IVFSQ4 index for large indexes
make index-large DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_dev_wiki/dump/
```

For `dev_wiki_noise` and `20181220_concat`, you need to modify the number of clusters to 101,372 and 1,048,576, respectively, and also use `index-add` and `index-merge` to add phrase representations to the index (see `Makefile` for details). If you want to use IVFPQ, using `index-large-pq` is enough in any case.

For evaluating the performance of DensePhrases on these larger phrase indexes, use `eval-dump`.
```bash
# Evaluate on the NQ development set questions
make eval-dump MODEL_NAME=dph-nqsqd-pb2 DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_dev_wiki/dump/
```

Optionally, you may want to compress the metadata (phrase dumps saved as hdf5 files) for a faster inference by loading it on RAM. This is only supported for the PQ index.
```bash
# Compress metadata of the entire Wikipedia (20181220_concat)
make compress-meta DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_20181220_concat/dump
```

### 3. Query-side fine-tuning
With a single 11GB GPU, you can easily train a query encoder to retrieve phrase-level knowledge from Wikipedia. First, you need a phrase index for the full Wikipedia (`20181220_concat`), which can be obtained by simply downloading from [here](#3-phrase-index) (`dph-nqsqd-pb2_20181220_concat`) or by creating a custom phrase index as described above.

The following command query-side fine-tunes `dph-nqsqd-pb2` on NQ.
```bash
# Query-side fine-tune on Natural Questions (model will be saved as MODEL_NAME)
make train-query MODEL_NAME=dph-nqsqd-pb2-nq DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_20181220_concat/dump/
```
Note that the pre-trained encoder is specified in `train-query` as `--query_encoder_path $(DPH_SAVE_DIR)/dph-nqsqd-pb2` and a new model will be saved as `dph-nqsqd-pb2-nq` as specified above. You can also train on different datasets by changing the dependency `nq-open-data` to `*-open-data` (e.g., `trec-open-data`).

#### IVFPQ vs IVFSQ4
Currently, `train-query` uses the IVFPQ index for query-side fine-tuning, and you should modify the arguments `--index_dir start/1048576_flat_PQ96_8` to `--index_dir start/1048576_flat_SQ4` for using IVFSQ4 index used in our paper.
For IVFPQ, training takes 2 to 3 hours per epoch for large datasets (NQ, TQA, SQuAD), and 3 to 8 minutes for small datasets (WQ, TREC). For IVFSQ4, the training time is highly dependent on the File I/O speed, so using SSDs is recommended for IVFSQ4.

### 4. Inference
With a pre-trained DensePhrases encoder (e.g., `dph-nqsqd-pb2_pq96-nq-10`) and a phrase index (e.g., `dph-nqsqd-pb2_20181220_concat`), you can test your queries as follows:

```bash
# Evaluate on Natural Questions
make eval-od MODEL_NAME=dph-nqsqd-pb2_pq96-nq-10 DUMP_DIR=$DPH_SAVE_DIR/dph-nqsqd-pb2_20181220_concat/dump/

# If the demo is being served on http://localhost:51997
make eval-od-req I_PORT=51997
```

## Pre-processing
At the bottom of `Makefile`, we list commands that we used for pre-processing the datasets and Wikipedia. For training question generation models (T5-large), we used [https://github.com/patil-suraj/question\_generation](https://github.com/patil-suraj/question_generation) (see also [here](https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/scripts/question_generation/generate_squad.py) for QG). Note that all datasets are already pre-processed including the generated questions, so you do not need to run most of these scripts. For creating test sets for custom (open-domain) questions, see `preprocess-openqa` in `Makefile`.

## Reference
Please cite our paper if you use DensePhrases in your work:
```bibtex
@article{lee2020learning,
  title={Learning Dense Representations of Phrases at Scale},
  author={Lee, Jinhyuk and Sung, Mujeen and Kang, Jaewoo and Chen, Danqi},
  journal={arXiv preprint arXiv:2012.12624},
  year={2020}
}
```

## License
Please see LICENSE for details.

[demo]: http://densephrases.korea.ac.kr
