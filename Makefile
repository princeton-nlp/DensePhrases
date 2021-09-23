############################## Single-passage Training + Normalization ###################################

model-name:
ifeq ($(MODEL_NAME),)
	echo "Please set MODEL_NAME before training (e.g., MODEL_NAME=test)"; exit 2;
endif

# Dataset paths for single-passage training (QG, train, dev, semi-od)
nq-rc-data:
	$(eval TRAIN_QG_DATA=nq/train_wiki3_na_filtered_qg_t5l35-sqd_filtered.json)
	$(eval TRAIN_DATA=nq/train_wiki3.json)
	$(eval DEV_DATA=nq/dev_wiki3.json)
	$(eval SOD_DATA=open-qa/nq-open/dev_wiki3_open.json)
	$(eval OPTIONS=--truecase)
sqd-rc-data:
	$(eval TRAIN_QG_DATA=squad/train-v1.1_qg_ents_t5large_3500_filtered.json)
	$(eval TRAIN_DATA=squad/train-v1.1.json)
	$(eval DEV_DATA=squad/dev-v1.1.json)
	$(eval SOD_DATA=open-qa/squad/test_preprocessed.json)
nqsqd-rc-data:
	$(eval TRAIN_QG_DATA=squad-nq/train-sqdqg_nqqg_filtered.json)
	$(eval TRAIN_DATA=squad-nq/train-sqd_nq.json)
	$(eval DEV_DATA=nq/dev_wiki3.json)
	$(eval SOD_DATA=open-qa/nq-open/dev_wiki3_open.json)
	$(eval OPTIONS=--truecase)
multi-rc-data:
	$(eval TRAIN_QG_DATA=squad-nq/train-sqdqg_nqqg_filtered.json)
	$(eval TRAIN_DATA=nq-wq-trec-tqa-sqd_train.json)
	$(eval DEV_DATA=nq/dev_wiki3.json)
	$(eval SOD_DATA=open-qa/nq-open/dev_wiki3_open.json)
	$(eval OPTIONS=--truecase)
paq-rc-data:
	$(eval TRAIN_DATA=paq/PAQ.metadata.jsonl)
	$(eval DEV_DATA=nq/dev_wiki3.json)
	$(eval SOD_DATA=open-qa/nq-open/dev_wiki3_open.json)
	
# Choose hyperparameter
pbn-param:
	$(eval PBN_OPTIONS=--pbn_size 2 --pbn_tolerance 0)
nq-param:
	$(eval BS=48)
	$(eval LR=3e-5)
	$(eval MAX_SEQ_LEN=192)
	$(eval LAMBDA_KL=2.0)
	$(eval LAMBDA_NEG=4.0)
	$(eval TEACHER_NAME=spanbert-base-cased-nq)
sqd-param:
	$(eval BS=24)
	$(eval LR=3e-5)
	$(eval MAX_SEQ_LEN=384)
	$(eval LAMBDA_KL=4.0)
	$(eval LAMBDA_NEG=2.0)
	$(eval TEACHER_NAME=spanbert-base-cased-squad)
nqsqd-param:
	$(eval BS=48)
	$(eval LR=3e-5)
	$(eval MAX_SEQ_LEN=192)
	$(eval LAMBDA_KL=4.0)
	$(eval LAMBDA_NEG=4.0)
	$(eval TEACHER_NAME=spanbert-base-cased-sqdnq)

# Choose index size
small-index:
	$(eval NUM_CLUSTERS=256)
	$(eval INDEX_TYPE=OPQ96)
medium1-index:
	$(eval NUM_CLUSTERS=16384)
	$(eval INDEX_TYPE=OPQ96)
medium2-index:
	$(eval NUM_CLUSTERS=131072)
	$(eval INDEX_TYPE=OPQ96)
large-index:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=OPQ96)
large-index-sq:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_TYPE=SQ4)

# Followings are template commands. See 'run-rc-nq' for a detailed use.
# 1) Training phrase and question encoders on reading comprehension.
train-rc: model-name nq-rc-data nq-param
	python train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DATA_DIR)/single-qa \
		--cache_dir $(CACHE_DIR) \
		--train_file $(TRAIN_DATA) \
		--predict_file $(DEV_DATA) \
		--do_train \
		--do_eval \
		--fp16 \
		--per_gpu_train_batch_size $(BS) \
		--learning_rate $(LR) \
		--num_train_epochs 2.0 \
		--max_seq_length $(MAX_SEQ_LEN) \
		--lambda_kl $(LAMBDA_KL) \
		--lambda_neg $(LAMBDA_NEG) \
		--lambda_flt 1.0 \
		--filter_threshold -2.0 \
		--append_title \
		--evaluate_during_training \
		--teacher_dir $(SAVE_DIR)/$(TEACHER_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--overwrite_output_dir \
		$(OPTIONS)

# 1-1) Sams as train-rc but with DDP
train-rc-ddp:
	OMP_NUM_THREADS=20 python -m torch.distributed.launch \
		--nnode=1 --node_rank=0 --nproc_per_node=4 train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DATA_DIR)/single-qa \
		--cache_dir $(CACHE_DIR) \
		--train_file $(TRAIN_DATA) \
		--predict_file $(DEV_DATA) \
		--do_train \
		--do_eval \
		--fp16 \
		--per_gpu_train_batch_size $(BS) \
		--learning_rate $(LR) \
		--num_train_epochs 2.0 \
		--max_seq_length $(MAX_SEQ_LEN) \
		--lambda_kl $(LAMBDA_KL) \
		--lambda_neg $(LAMBDA_NEG) \
		--lambda_flt 1.0 \
		--filter_threshold -2.0 \
		--append_title \
		--evaluate_during_training \
		--teacher_dir $(SAVE_DIR)/$(TEACHER_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		$(OPTIONS)

# 2) Trained phrase encoders can be used to generate phrase vectors
gen-vecs:
	python generate_phrase_vecs.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DATA_DIR)/single-qa \
		--cache_dir $(CACHE_DIR) \
		--predict_file $(DEV_DATA) \
		--do_dump \
		--max_seq_length 512 \
		--doc_stride 500 \
		--fp16 \
		--filter_threshold -2.0 \
		--append_title \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		$(OPTIONS)

# 3) Build an IVFOPQ index for generated phrase vectors
index-vecs: dump-dir large-index
	python build_phrase_index.py \
		--dump_dir $(DUMP_DIR) \
		--stage all \
		--replace \
		--num_clusters $(NUM_CLUSTERS) \
		--fine_quant $(INDEX_TYPE) \
		--cuda

# 4) Compress metadata
compress-meta:
	python scripts/preprocess/compress_metadata.py \
		--input_dump_dir $(DUMP_DIR)/phrase \
		--output_dir $(DUMP_DIR)

# 5) Evaluate the phrase index for phrase retrieval
eval-index: dump-dir model-name large-index nq-open-data
	python eval_phrase_retrieval.py \
		--run_mode eval \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--save_pred \
		--aggregate \
		$(OPTIONS)

# Sample usage (If this runs without an error, you are all set!)
draft: model-name nq-rc-data nq-param pbn-param small-index
	make train-rc \
		TRAIN_DATA=$(TRAIN_DATA) DEV_DATA=$(DEV_DATA) \
		TEACHER_NAME=$(TEACHER_NAME) MODEL_NAME=$(MODEL_NAME) \
		BS=$(BS) LR=$(LR) MAX_SEQ_LEN=$(MAX_SEQ_LEN) \
		LAMBDA_KL=$(LAMBDA_KL) LAMBDA_NEG=$(LAMBDA_NEG) \
		OPTIONS='$(PBN_OPTIONS) --draft'
	make gen-vecs \
		DEV_DATA=$(DEV_DATA) MODEL_NAME=$(MODEL_NAME)
	make index-vecs \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE)
	make compress-meta \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump
	make eval-index \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE) \
		MODEL_LANE=$(MODEL_NAME) TEST_DATA=$(SOD_DATA) \
		OPTIONS=$(OPTIONS)

# Single-passage training + additional negatives for NQ
# Available datasets: NQ (nq-rc-data), SQuAD (sqd-rc-data), NQ+SQuAD (nqsqd-rc-data)
# Should change hyperparams (e.g., nq-param) accordingly
run-rc-nq: model-name nq-rc-data nq-param pbn-param small-index
	make train-rc \
		TRAIN_DATA=$(TRAIN_QG_DATA) DEV_DATA=$(DEV_DATA) \
		TEACHER_NAME=$(TEACHER_NAME) MODEL_NAME=$(MODEL_NAME)_tmp \
		BS=$(BS) LR=$(LR) MAX_SEQ_LEN=$(MAX_SEQ_LEN) \
		LAMBDA_KL=$(LAMBDA_KL) LAMBDA_NEG=$(LAMBDA_NEG)
	make train-rc \
		TRAIN_DATA=$(TRAIN_DATA) DEV_DATA=$(DEV_DATA) \
		TEACHER_NAME=$(TEACHER_NAME) MODEL_NAME=$(MODEL_NAME) \
		BS=$(BS) LR=$(LR) MAX_SEQ_LEN=$(MAX_SEQ_LEN) \
		LAMBDA_KL=$(LAMBDA_KL) LAMBDA_NEG=$(LAMBDA_NEG) \
		OPTIONS='$(PBN_OPTIONS) --load_dir $(SAVE_DIR)/$(MODEL_NAME)_tmp'
	make gen-vecs \
		DEV_DATA=$(DEV_DATA) MODEL_NAME=$(MODEL_NAME)
	make index-vecs \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE)
	make compress-meta \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump
	make eval-index \
		DUMP_DIR=$(SAVE_DIR)/$(MODEL_NAME)/dump \
		NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE) \
		MODEL_LANE=$(MODEL_NAME) TEST_DATA=$(SOD_DATA) \
		OPTIONS=$(OPTIONS)

# Testing filter thresholds
filter-test: model-name nq-rc-data
	python train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DATA_DIR)/single-qa \
		--cache_dir $(CACHE_DIR) \
		--predict_file $(DEV_DATA) \
		--do_filter_test \
		--append_title \
		--filter_threshold_list " -4,-3,-2,-1,-0.5,0,1" \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \

# Training cross encoder
train-cross: model-name nq-rc-data
	python train_cross_encoder.py \
		--model_type bert \
		--model_name_or_path SpanBERT/spanbert-large-cased \
		--do_train \
		--do_eval \
		--cache_dir $(CACHE_DIR) \
		--train_file $(DATA_DIR)/single-qa/$(TRAIN_DATA) \
		--predict_file $(DATA_DIR)/single-qa/$(DEV_DATA) \
		--per_gpu_train_batch_size 8 \
		--learning_rate 1e-5 \
		--num_train_epochs 2.0 \
		--max_seq_length 384 \
		--doc_stride 128 \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME)

############################## Large-scale Dump & Indexing ###############################

dump-dir:
ifeq ($(DUMP_DIR),)
	echo "Please set DUMP_DIR before dumping/indexing (e.g., DUMP_DIR=test)"; exit 2;
endif

# Wikipedia dumps (specified as 'data_name') in diffent sizes and their recommended number of clusters for IVF
# - wiki-dev: 1/100 Wikpedia scale (sampled), num_clusters=16384 (medium1-index)
# - wiki-dev-noise: 1/10 Wikipedia scale (sampled), num_clusters=131072 (medium2-index)
# - wiki-20181220: full Wikipedia scale, num_clusters=1048576 (large-index)

# Dump phrase vectors in parallel. Dump will be saved in $(SAVE_DIR)/$(MODEL_NAME)_(data_name)/dump.
gen-vecs-parallel: model-name
	nohup python scripts/parallel/dump_phrases.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(CACHE_DIR) \
		--data_dir $(DATA_DIR)/wikidump \
		--data_name wiki-dev \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--filter_threshold 1.0 \
		--append_title \
		--start $(START) \
		--end $(END) \
		> $(SAVE_DIR)/logs/$(MODEL_NAME)_$(START)-$(END).log &

# Parallel add for large-scale on-disk IVFSQ (start, end = file idx)
index-add: dump-dir large-index-sq
	export MKL_SERVICE_FORCE_INTEL=1
	python scripts/parallel/add_to_index.py \
		--dump_dir $(DUMP_DIR) \
		--num_clusters $(NUM_CLUSTERS) \
		--cuda \
		--start $(START) \
		--end $(END)

# Merge for large-scale on-disk IVFSQ
index-merge: dump-dir large-index-sq
	python build_phrase_index.py \
		--dump_dir $(DUMP_DIR) \
		--stage merge \
		--replace \
		--num_clusters $(NUM_CLUSTERS) \
		--fine_quant $(INDEX_TYPE)

############################## Open-domain Search & Query-side Fine-tuning ###################################

# Dataset paths for open-domain QA and slot filling (with options)
nq-open-data:
	$(eval TRAIN_DATA=open-qa/nq-open/train_preprocessed.json)
	$(eval DEV_DATA=open-qa/nq-open/dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/nq-open/test_preprocessed.json)
	$(eval OPTIONS=--truecase)
wq-open-data:
	$(eval TRAIN_DATA=open-qa/webq/WebQuestions-train-nodev_preprocessed.json)
	$(eval DEV_DATA=open-qa/webq/WebQuestions-dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/webq/WebQuestions-test_preprocessed.json)
	$(eval OPTIONS=--truecase --candidate_path $(DATA_DIR)/open-qa/webq/freebase-entities.txt)
trec-open-data:
	$(eval TRAIN_DATA=open-qa/trec/CuratedTrec-train-nodev_preprocessed.json)
	$(eval DEV_DATA=open-qa/trec/CuratedTrec-dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/trec/CuratedTrec-test_preprocessed.json)
	$(eval OPTIONS=--regex)
tqa-open-data:
	$(eval TRAIN_DATA=open-qa/triviaqa-unfiltered/train_preprocessed.json)
	$(eval DEV_DATA=open-qa/triviaqa-unfiltered/dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/triviaqa-unfiltered/test_preprocessed.json)
sqd-open-data:
	$(eval TRAIN_DATA=open-qa/squad/train_preprocessed.json)
	$(eval DEV_DATA=open-qa/squad/dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/squad/test_preprocessed.json)
nq-fid-data:
	$(eval TRAIN_DATA=open-qa/nq-fid/train_preprocessed.json)
	$(eval DEV_DATA=open-qa/nq-fid/dev_preprocessed.json)
	$(eval TEST_DATA=open-qa/nq-fid/test_preprocessed.json)
	$(eval OPTIONS=--truecase)
kilt-options:
	$(eval OPTIONS=--is_kilt --title2wikiid_path $(DATA_DIR)/wikidump/title2wikiid.json)
trex-kilt-data: kilt-options
	$(eval TRAIN_DATA=kilt/trex/trex-train-kilt_open_10000.json)
	$(eval DEV_DATA=kilt/trex/trex-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/trex/trex-test-kilt_open.json)
	$(eval OPTIONS=$(OPTIONS) --kilt_gold_path $(DATA_DIR)/kilt/trex/trex-dev-kilt.jsonl --agg_strat opt2)
zsre-kilt-data: kilt-options
	$(eval TRAIN_DATA=kilt/zsre/structured_zeroshot-train-kilt_open_10000.json)
	$(eval DEV_DATA=kilt/zsre/structured_zeroshot-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/zsre/structured_zeroshot-test-kilt_open.json)
	$(eval OPTIONS=$(OPTIONS) --kilt_gold_path $(DATA_DIR)/kilt/zsre/structured_zeroshot-dev-kilt.jsonl --agg_strat opt2)
ay2-kilt-data: kilt-options
	$(eval TRAIN_DATA=kilt/ay2/aidayago2-train-kilt_open.json)
	$(eval DEV_DATA=kilt/ay2/aidayago2-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/ay2/aidayago2-test_without_answers-kilt-fake_open.json)
	$(eval OPTIONS=$(OPTIONS) --truecase --kilt_gold_path $(DATA_DIR)/kilt/ay2/aidayago2-dev-kilt.jsonl --agg_strat opt2 --max_query_length 384 --label_strat doc)
cweb-kilt-data: kilt-options
	$(eval DEV_DATA=kilt/cweb/cweb-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/cweb/cweb-test_without_answers-kilt-fake_open.json)
	$(eval OPTIONS=$(OPTIONS) --truecase --kilt_gold_path $(DATA_DIR)/kilt/cweb/cweb-dev-kilt.jsonl --agg_strat opt2 --max_query_length 384)
wned-kilt-data: kilt-options
	$(eval DEV_DATA=kilt/wned/wned-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/wned/wned-test_without_answers-kilt-fake_open.json)
	$(eval OPTIONS=$(OPTIONS) --truecase --kilt_gold_path $(DATA_DIR)/kilt/wned/wned-dev-kilt.jsonl --agg_strat opt2 --max_query_length 384)
wow-kilt-data: kilt-options
	$(eval TRAIN_DATA=kilt/wow/wow-train-kilt_open.json)
	$(eval DEV_DATA=kilt/wow/wow-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/wow/wow-dev-kilt_open.json)
	$(eval OPTIONS=$(OPTIONS) --truecase --kilt_gold_path $(DATA_DIR)/kilt/wow/wow-dev-kilt.jsonl --agg_strat opt2 --max_query_length 384 --label_strat doc)
all-kilt-data: kilt-options
	$(eval TRAIN_DATA=kilt/kilt-combined-train-15k_open.json)
	$(eval DEV_DATA=kilt/kilt-combined-dev-2k_open.json)
	$(eval TEST_DATA=kilt/kilt-combined-dev-2k_open.json)
	$(eval OPTIONS=$(OPTIONS) --truecase --kilt_gold_path $(DATA_DIR)/kilt/kilt-combined-dev-2k.jsonl --agg_strat opt2 --max_query_length 384 --label_strat dynamic)

benchmark-data:
	$(eval TEST_DATA=scripts/benchmark/data/nq_1000_dev_denspi.json)
all-open-data:
	$(eval TEST_DATA=$(DATA_DIR)/open-qa/nq-open/test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/webq/WebQuestions-test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/trec/CuratedTrec-test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/triviaqa-unfiltered/test_preprocessed.json)
	$(eval TEST_DATA=$(TEST_DATA),$(DATA_DIR)/open-qa/squad/test_preprocessed.json)
	$(eval OPTIONS=--truecase)

# Query-side fine-tuning
train-query: dump-dir model-name nq-open-data large-index
	python train_query.py \
		--run_mode train_query \
		--cache_dir $(CACHE_DIR) \
		--train_path $(DATA_DIR)/$(TRAIN_DATA) \
		--dev_path $(DATA_DIR)/$(DEV_DATA) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--per_gpu_train_batch_size 12 \
		--eval_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 5 \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/densephrases-multi \
		--output_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--top_k 100 \
		--cuda \
		--save_pred \
		$(OPTIONS)

# Evalute all datasets
eval-index-all: dump-dir model-name large-index all-open-data
	python eval_phrase_retrieval.py \
		--run_mode eval_all \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(TEST_DATA) \
		--aggregate \
		$(OPTIONS)

# Post-processing KILT predictions
trex-gold:
	$(eval GOLD_FILE=$(DATA_DIR)/kilt/trex/trex-dev-kilt.jsonl)
zsre-gold:
	$(eval GOLD_FILE=$(DATA_DIR)/kilt/zsre/structured_zeroshot-dev-kilt.jsonl)
strip-kilt: zsre-gold
	python scripts/kilt/strip_pred.py \
		$(INPUT_PRED) \
		$(GOLD_FILE)	

################################ Demo Serving ###################################

# Serve question encoder
q-serve:
	nohup python run_demo.py \
		--run_mode q_serve \
		--cache_dir $(CACHE_DIR) \
		--load_dir princeton-nlp/$(MODEL_NAME) \
		--cuda \
		--max_query_length 32 \
		--query_port $(Q_PORT) > $(SAVE_DIR)/logs/q-serve_$(Q_PORT).log &

# Serve phrase index (Q_PORT may change)
p-serve: dump-dir large-index
	nohup python run_demo.py \
		--run_mode p_serve \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--cuda \
		--truecase \
		--dump_dir $(DUMP_DIR) \
		--query_port $(Q_PORT) \
		--index_port $(I_PORT) > $(SAVE_DIR)/logs/p-serve_$(I_PORT).log &

# Evaluation using the open QA demo (used for benchmark)
eval-demo: nq-open-data
	python run_demo.py \
		--run_mode eval_request \
		--index_port $(I_PORT) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--eval_batch_size 64 \
		--save_pred \
		$(OPTIONS)

# (Optional) Serve single-passage RC demo
single-serve:
	nohup python run_demo.py \
		--run_mode single_serve \
		--cuda \
		--cache_dir $(CACHE_DIR) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--query_port $(Q_PORT) > $(SAVE_DIR)/logs/s-serve_$(Q_PORT).log &

############################## Passage-level evaluation ###################################

# agg_strat=opt2 means passage retrieval
eval-index-psg: dump-dir model-name large-index nq-open-data
	python eval_phrase_retrieval.py \
		--run_mode eval \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--dump_dir $(DUMP_DIR) \
		--index_name start/$(NUM_CLUSTERS)_flat_$(INDEX_TYPE) \
		--load_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DATA_DIR)/$(TEST_DATA) \
		--save_pred \
		--aggregate \
		--agg_strat opt2 \
		--top_k 200 \
		--eval_psg \
		--psg_top_k 100 \
		$(OPTIONS)

# transform prediction for the recall evaluation (when you already have prediction files)
recall-eval: model-name
	python scripts/postprocess/recall_transform.py \
		--model_dir $(SAVE_DIR)/$(MODEL_NAME) \
		--pred_file $(PRED_NAME).pred \
		--psg_top_k 100
	python scripts/postprocess/recall.py \
		--k_values 1,5,20,100 \
		--results_file $(SAVE_DIR)/$(MODEL_NAME)/pred/$(PRED_NAME)_psg-top100.json \
		--ans_fn string

############################## Data Pre/Post-processing ###################################

preprocess-openqa:
	python scripts/preprocess/create_openqa.py \
		$(FS)/fid-data/download/NQ-open.train.jsonl \
		$(DATA_DIR)/open-qa/nq-new \
		--input_type jsonl

# Warning: many scripts below are not documented well.
# Each script may rely on external resources (e.g., original NQ datasets).
data-config:
	$(eval NQORIG_DIR=$(DATA_DIR)/natural-questions)
	$(eval NQOPEN_DIR=$(DATA_DIR)/nq-open)
	$(eval NQREADER_DIR=$(DATA_DIR)/single-qa/nq)
	$(eval SQUAD_DIR=$(DATA_DIR)/single-qa/squad)
	$(eval SQUADREADER_DOC_DIR=$(DATA_DIR)/squad-reader-docs)
	$(eval NQREADER_DOC_DIR=$(DATA_DIR)/nq-reader-docs)
	$(eval WIKI_DIR=$(DATA_DIR)/wikidump)

nq-reader-to-wiki:
	python scripts/preprocess/create_nq_reader_wiki.py \
		$(DATA_DIR)/single-qa/nq/train.json,$(DATA_DIR)/single-qa/nq/dev.json \
		$(DATA_DIR)/single-qa/nq \
		$(DATA_DIR)/wikidump/20181220_concat/

download-wiki: data-config
	python scripts/preprocess/download_wikidump.py \
		--output_dir $(WIKI_DIR)

nq-reader-train: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_path $(NQREADER_DIR)/train_79168.json

nq-reader-dev: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_8757.json

nq-reader-dev-sample: data-config
	python scripts/preprocess/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_sample.json

nq-reader-train-docs: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_dir $(NQREADER_DOC_DIR)/train

nq-reader-dev-docs: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)/dev

nq-reader-dev-docs-sample: data-config
	python scripts/preprocess/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)-sample

nq-_reader-train-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/train \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki

nq-reader-dev-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/dev \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki

squad-reader-train-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/train-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/train_wiki \
		--is_squad

squad-reader-dev-docs-wiki: data-config
	python scripts/preprocess/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/dev-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/dev_wiki \
		--is_squad

build-db: data-config
	python scripts/preprocess/build_db.py \
		--data_path $(WIKI_DIR)/extracted \
		--save_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--preprocess scripts/preprocess/prep_wikipedia.py \
		--overwrite

build-wikisquad: data-config
	python scripts/preprocess/build_wikisquad.py \
		--db_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--out_dir $(WIKI_DIR)/20181220_nolist

concat-wikisquad: data-config
	python scripts/preprocess/concat_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_concat

first-para-wikisquad: data-config
	python scripts/preprocess/first_para_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_first

compare-db: data-config
	python scripts/preprocess/compare_db.py \
		--db1 $(DATA_DIR)/denspi/docs.db \
		--db2 $(WIKI_DIR)/docs_20181220.db
		
sample-nq-reader-doc-wiki-train: data-config
	python scripts/preprocess/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.15 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/train_wiki \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki_noise

sample-nq-reader-doc-wiki-dev: data-config
	python scripts/preprocess/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.1 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/dev_wiki \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki_noise
