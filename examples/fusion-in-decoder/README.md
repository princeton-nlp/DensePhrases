# Fusion-in-Decoder with DensePhrases
You can use retrieved passages from DensePhrases to build a state-of-the-art open-domain QA system called [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) (FiD).
Note that DensePhrases (w/o reader) already provides phrase-level answers for end-to-end open-domain QA whose performance is comparable to DPR (w/ BERT reader). This section provides how you can further improve the performance using a generative reader model (T5).

## Getting Top Passages from DensePhrases
First, you need to get passages from DensePhrases.
Using DensePhrases-multi, you can retrieve passages for Natural Questions as follows:
```
TRAIN_DATA=open-qa/nq-open/train_preprocessed.json
DEV_DATA=open-qa/nq-open/dev_preprocessed.json
TEST_DATA=open-qa/nq-open/test_preprocessed.json

# Change --test_path accordingly
python eval_phrase_retrieval.py \
    --run_mode eval \
    --model_type bert \
    --pretrained_name_or_path SpanBERT/spanbert-base-cased \
    --cuda \
    --dump_dir $SAVE_DIR/densephrases-multi_wiki-20181220/dump/ \
    --index_name start/1048576_flat_OPQ96 \
    --load_dir $SAVE_DIR/densephrases-multi-query-nq \
    --test_path $DATA_DIR/$TEST_DATA \
    --save_pred \
    --aggregate \
    --agg_strat opt2 \
    --top_k 200 \
    --eval_psg \
    --psg_top_k 100 \
    --truecase
```
Since FiD requires training passages, you need to change `--test_path` to `$TRAIN_DATA` or `$DEV_DATA` to get training or development passages, respectively.
Equivalently, you can use `eval-index-psg` in our [Makefile](https://github.com/princeton-nlp/DensePhrases/blob/main/Makefile).
For TriviaQA, simply change the dataset to `tqa-open-data` specified in Makefile.

After the inference, you will be able to get the following three files used for training and evaluating FiD models:
* train_preprocessed_79168_top200_psg-top100.json
* dev_preprocessed_8757_top200_psg-top100.json
* test_preprocessed_3610_top200_psg-top100.json

We will assume that these files are saved under `$SAVE_DIR/fid-data`.
Note that each retrieved passage in DensePhrases is a natural paragraph mostly in different lengths. For the exact replication of the experiments in our EMNLP paper, you need a phrase index created from Wikipedia pre-processed for DPR (100-word passages), which we plan to provide soonish.

## Installing Fusion-in-Decoder
For Fusion-in-Decoder, we use [the official code](https://github.com/facebookresearch/FiD) provided by the authors.
It is often better to use a separate conda environment to train FiD.
See [here](https://github.com/facebookresearch/FiD#dependencies) for dependencies.

```bash
# Install torch with conda (please check your CUDA version)
conda create -n fid python=3.7
conda activate fid
conda install pytorch=1.9.0 cudatoolkit=11.0 -c pytorch

# Install Fusion-in-Decoder
git clone https://github.com/facebookresearch/FiD.git
cd FiD
pip install -r requirements.txt
```

## Training and Evaluation
```bash
TRAIN_DATA=fid-data/train_preprocessed_79168_top200_psg-top100.json
DEV_DATA=fid-data/dev_preprocessed_8757_top200_psg-top100.json
TEST_DATA=fid-data/test_preprocessed_3610_top200_psg-top100.json

# Train T5-base with top 5 passages (DDP using 4 GPUs)
nohup python /path/to/miniconda3/envs/fid/lib/python3.6/site-packages/torch/distributed/launch.py \
    --nnode=1 --node_rank=0 --nproc_per_node=4 train_reader.py \
    --train_data $SAVE_DIR/$TRAIN_DATA \
    --eval_data $SAVE_DIR/$DEV_DATA \
    --model_size base \
    --per_gpu_batch_size 1 \
    --accumulation_steps 16 \
    --total_steps 160000 \
    --eval_freq 8000 \
    --save_freq 8000 \
    --n_context 5 \
    --lr 0.00005 \
    --text_maxlength 300 \
    --name nq_reader_base-dph-c5-d4 \
    --checkpoint_dir $SAVE_DIR/fid-data/pretrained_models > nq_reader_base-dph-c5-d4_out.log &

# Test T5-base with top 5 passages (DDP using 4 GPUs)
python /n/fs/nlp-jl5167/miniconda3/envs/fid/lib/python3.6/site-packages/torch/distributed/launch.py \
    --nnode=1 --node_rank=0 --nproc_per_node=4 test_reader.py \
    --model_path $SAVE_DIR/fid-data/pretrained_models/nq_reader_base-dph-c5-d4/checkpoint/best_dev \
    --eval_data $SAVE_DIR/$TEST_DATA \
    --per_gpu_batch_size 1 \
    --n_context 100 \
    --write_results \
    --name nq_reader_base-dph-c5-d4 \
    --checkpoint_dir $SAVE_DIR/fid-data/pretrained_models \
    --text_maxlength 300
```
Note that most hyperparameters follow the original work and the only difference is the use of `--accumulation_steps 16` and proper adjustment to its training, save, evaluation steps. Larger `--text_maxlength` is used to cover natural paragraphs that are often longer than 100 words.
