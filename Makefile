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
	
# Choose hyperparameter
pbn-param:
	$(eval PBN_OPTIONS=--pbn_size 2 --pbn_tolerance 0)
nq-param:
	$(eval BS=64)
	$(eval LR=1e-4)
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
	$(eval BS=64)
	$(eval LR=1e-4)
	$(eval MAX_SEQ_LEN=192)
	$(eval LAMBDA_KL=4.0)
	$(eval LAMBDA_NEG=4.0)
	$(eval TEACHER_NAME=spanbert-base-cased-sqdnq)

# Choose index type
small-index:
	$(eval NUM_CLUSTERS=256)
	$(eval INDEX_NAME=OPQ96)
medium1-index:
	$(eval NUM_CLUSTERS=16384)
	$(eval INDEX_NAME=OPQ96)
medium2-index:
	$(eval NUM_CLUSTERS=131072)
	$(eval INDEX_NAME=OPQ96)
large-index:
	$(eval NUM_CLUSTERS=1048576)
	$(eval INDEX_NAME=OPQ96)

# Followings are template commands. See 'train-rc-nq' for a detailed use.
# 1) Training phrase and question encoders on reading comprehension.
train-rc:
	python train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DPH_DATA_DIR)/single-qa \
		--cache_dir $(DPH_CACHE_DIR) \
		--train_file $(TRAIN_DATA) \
		--predict_file $(DEV_DATA) \
		--do_train \
		--do_eval \
		--per_gpu_train_batch_size $(BS) \
		--learning_rate $(LR) \
		--num_train_epochs 2.0 \
		--max_seq_length $(MAX_SEQ_LEN) \
		--fp16 \
		--lambda_kl $(LAMBDA_KL) \
		--lambda_neg $(LAMBDA_NEG) \
		--lambda_flt 1.0 \
		--filter_threshold -2.0 \
		--append_title \
		--evaluate_during_training \
		--teacher_dir $(DPH_SAVE_DIR)/$(TEACHER_NAME) \
		--output_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		$(OPTIONS)

# 2) Trained phrase encoders can be used to generate phrase vectors
gen-vecs:
	python generate_phrase_vecs.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DPH_DATA_DIR)/single-qa \
		--cache_dir $(DPH_CACHE_DIR) \
		--predict_file $(DEV_DATA) \
		--do_dump \
		--max_seq_length 512 \
		--doc_stride 500 \
		--fp16 \
		--filter_threshold -2.0 \
		--append_title \
		--load_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		$(OPTIONS)

# 3) Build an IVFOPQ index for generated phrase vectors
index-vecs:
	python build_phrase_index.py \
		$(DPH_SAVE_DIR)/$(MODEL_NAME)/dump all \
		--replace \
		--num_clusters $(NUM_CLUSTERS) \
		--fine_quant $(INDEX_NAME) \
		--cuda

# 4) Evaluate the phrase index for phrase retrieval
eval-index: model-name
	python -m densephrases.experiments.run_open \
		--run_mode eval_inmemory \
		--cuda \
		--dump_dir $(DPH_SAVE_DIR)/$(MODEL_NAME)/dump \
		--index_dir start/$(NUM_CLUSTERS)_flat_$(INDEX_NAME) \
		--query_encoder_path $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DPH_DATA_DIR)/$(EVAL_DATA) \
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
	make index-vecs NUM_CLUSTERS=$(NUM_CLUSTERS) INDEX_TYPE=$(INDEX_TYPE)
	make eval-index EVAL_DATA=$(SOD_DATA) OPTIONS=$(OPTIONS)

# Single-passage training + additional negatives for NQ (simply change 'nq' to 'sqd' for SQuAD)
train-rc-nq: model-name nq-rc-data nq-param pbn-param
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
		OPTIONS='$(PBN_OPTIONS) --do_dump --load_dir $(DPH_SAVE_DIR)/$(MODEL_NAME)_tmp'
	make index-sod
	make eval-sod SOD_DATA=$(SOD_DATA) OPTIONS=$(OPTIONS)

# Test filter thresholds
filter-test: model-name nq-rc-data
	python train_rc.py \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--data_dir $(DPH_DATA_DIR)/single-qa \
		--cache_dir $(DPH_CACHE_DIR) \
		--predict_file $(DEV_DATA) \
		--do_filter_test \
		--append_title \
		--filter_threshold_list " -4,-3,-2,-1,-0.5,0" \
		--load_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--draft

############################## Large-scale Dump & Indexing ###############################

dump-dir:
ifeq ($(DUMP_DIR),)
	echo "Please set DUMP_DIR before dumping/indexing (e.g., DUMP_DIR=test)"; exit 2;
endif

# Wikipedia dumps in diffent sizes and their recommended number of clusters for IVF
# - dev_wiki: 1/100 Wikpedia scale (sampled), num_clusters=16384
# - dev_wiki_noise: 1/10 Wikipedia scale (sampled), num_clusters=131072
# - 20181220_concat: full Wikipedia scale, num_clusters=1048576

# Dump phrase reps in parallel. Dump will be saved in $(DPH_SAVE_DIR)/$(MODEL_NAME)_(data_name)/dump.
# Please move the dump to an SSD $(DUMP_DIR) for a faster indexing.
dump-large: model-name
	nohup python -m densephrases.experiments.parallel.dump_phrases \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(DPH_CACHE_DIR) \
		--data_dir $(DPH_DATA_DIR)/wikidump \
		--data_name dev_wiki \
		--load_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--output_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--filter_threshold 0.0 \
		--append_title \
		--start $(START) \
		--end $(END) \
		> $(DPH_SAVE_DIR)/logs/$(MODEL_NAME)_$(START)-$(END).log &

# IVFSQ indexing: for 131072 or 1048576, stop the process after training the quantizer and use index-add below
index-large: dump-dir
	python -m densephrases.experiments.create_index \
		$(DUMP_DIR) all \
		--replace \
		--num_clusters 16384 \
		--fine_quant SQ4 \
		--cuda

# Parallel add for large-scale on-disk IVFSQ (start, end = file idx)
index-add: dump-dir
	export MKL_SERVICE_FORCE_INTEL=1
	python -m densephrases.experiments.parallel.add_to_index \
		--dump_dir $(DUMP_DIR) \
		--num_clusters 1048576 \
		--cuda \
		--start $(START) \
		--end $(END)

# Merge for large-scale on-disk IVFSQ
index-merge: dump-dir
	python -m densephrases.experiments.create_index \
	$(DUMP_DIR) merge \
	--num_clusters 1048576 \
	--replace \
	--fine_quant SQ4

# IVFPQ indexing (currently do not support parallelized add/merge)
index-large-pq: dump-dir
	python -m densephrases.experiments.create_index \
		$(DUMP_DIR) all \
		--replace \
		--num_clusters 1048576 \
		--fine_quant PQ96_8 \
		--cuda

# Use if for large-scale dump evaluation
eval-dump: model-name dump-dir nq-rc-data
	python -m densephrases.experiments.run_open \
		--run_mode eval_inmemory \
		--cuda \
		--dump_dir $(DUMP_DIR) \
		--index_dir start/16384_flat_SQ4 \
		--query_encoder_path $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DPH_DATA_DIR)/$(SOD_DATA) \
		$(OPTIONS)

# Compressed metadata to load it on RAM (only use for PQ)
compress-meta: dump-dir
	python -m densephrases.scripts.preprocess.compress_metadata \
		--input_dump_dir $(DUMP_DIR)/phrase \
		--output_dir $(DUMP_DIR)

############################## Open-domain Search & Query-side Fine-tuning ###################################

# Useful when runnig multiple query-side fine-tuning on the same server
limit-threads:
	$(eval OMP_NUM_THREADS=5)

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
	$(eval OPTIONS=--truecase --candidate_path $(DPH_DATA_DIR)/open-qa/webq/freebase-entities.txt)
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
kilt-options:
	$(eval OPTIONS=--is_kilt --title2wikiid_path $(DPH_DATA_DIR)/wikidump/title2wikiid.json)
trex-open-data: kilt-options
	$(eval TRAIN_DATA=kilt/trex/trex-train-kilt_open.json)
	$(eval DEV_DATA=kilt/trex/trex-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/trex/trex-dev-kilt_open.json)
	$(eval OPTIONS=$(OPTIONS) --kilt_gold_path $(DPH_DATA_DIR)/kilt/trex/trex-dev-kilt.jsonl)
zsre-open-data: kilt-options
	$(eval TRAIN_DATA=kilt/zsre/structured_zeroshot-train-kilt_open.json)
	$(eval DEV_DATA=kilt/zsre/structured_zeroshot-dev-kilt_open.json)
	$(eval TEST_DATA=kilt/zsre/structured_zeroshot-dev-kilt_open.json)
	$(eval OPTIONS=$(OPTIONS) --kilt_gold_path $(DPH_DATA_DIR)/kilt/zsre/structured_zeroshot-dev-kilt.jsonl)
benchmark-data:
	$(eval TEST_DATA=densephrases/scripts/benchmark/data/nq_1000_dev_denspi.json)

eval-od: dump-dir model-name nq-open-data
	python -m densephrases.experiments.run_open \
		--run_mode eval_inmemory \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--eval_batch_size 12 \
		--dump_dir $(DUMP_DIR) \
		--index_dir start/1048576_flat_PQ96_8 \
		--query_encoder_path $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--test_path $(DPH_DATA_DIR)/$(TEST_DATA) \
		$(OPTIONS)

train-query: dump-dir model-name nq-open-data
	python -m densephrases.experiments.run_open \
		--run_mode train_query \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(DPH_CACHE_DIR) \
		--train_path $(DPH_DATA_DIR)/$(TRAIN_DATA) \
		--dev_path $(DPH_DATA_DIR)/$(DEV_DATA) \
		--test_path $(DPH_DATA_DIR)/$(TEST_DATA) \
		--per_gpu_train_batch_size 12 \
		--eval_batch_size 12 \
		--learning_rate 3e-5 \
		--num_train_epochs 5 \
		--dump_dir $(DUMP_DIR) \
		--index_dir start/1048576_flat_PQ96_8 \
		--query_encoder_path $(DPH_SAVE_DIR)/dph-nqsqd-pb2 \
		--output_dir $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--top_k 100 \
		--cuda \
		$(OPTIONS)

################################ Demo Serving ###################################

# Serve question encoder
q-serve:
	nohup python -m densephrases.demo.serve \
		--run_mode q_serve \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cache_dir $(DPH_CACHE_DIR) \
		--query_encoder_path $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--cuda \
		--max_query_length 32 \
		--query_port $(Q_PORT) > $(DPH_SAVE_DIR)/logs/q-serve_$(Q_PORT).log &

# Serve phrase index (Q_PORT may change)
p-serve: dump-dir
	nohup python -m densephrases.demo.serve \
		--run_mode p_serve \
		--index_dir start/1048576_flat_PQ96_8 \
		--cuda \
		--truecase \
		--dump_dir $(DUMP_DIR) \
		--query_port $(Q_PORT) \
		--index_port $(I_PORT) > $(DPH_SAVE_DIR)/logs/p-serve_$(I_PORT).log &

# (Optional) Serve single-passage RC demo
single-serve:
	nohup python -m densephrases.demo.serve \
		--run_mode single_serve \
		--model_type bert \
		--pretrained_name_or_path SpanBERT/spanbert-base-cased \
		--cuda \
		--cache_dir $(DPH_CACHE_DIR) \
		--query_encoder_path $(DPH_SAVE_DIR)/$(MODEL_NAME) \
		--query_port $(Q_PORT) > $(DPH_SAVE_DIR)/logs/s-serve_$(Q_PORT).log &

# Evaluation using the open QA demo (used for benchmark)
eval-od-req: nq-open-data
	python -m densephrases.demo.serve \
		--run_mode eval_request \
		--index_port $(I_PORT) \
		--test_path $(DPH_DATA_DIR)/$(TEST_DATA) \
		--eval_batch_size 64 \
		$(OPTIONS)

############################## Data Pre/Post-processing ###################################

preprocess-openqa:
	python densephrases/scripts/preprocess/create_openqa.py \
		$(DPH_DATA_DIR)/single-qa/squad/train-v1.1.json \
		$(DPH_DATA_DIR)/open-qa/squad \
		--input_type SQuAD

# Warning: many scripts below are not documented well.
# Each script may rely on external resources (e.g., original NQ datasets).
data-config:
	$(eval NQORIG_DIR=$(DPH_DATA_DIR)/natural-questions)
	$(eval NQOPEN_DIR=$(DPH_DATA_DIR)/nq-open)
	$(eval NQREADER_DIR=$(DPH_DATA_DIR)/single-qa/nq)
	$(eval SQUAD_DIR=$(DPH_DATA_DIR)/single-qa/squad)
	$(eval SQUADREADER_DOC_DIR=$(DPH_DATA_DIR)/squad-reader-docs)
	$(eval NQREADER_DOC_DIR=$(DPH_DATA_DIR)/nq-reader-docs)
	$(eval WIKI_DIR=$(DPH_DATA_DIR)/wikidump)

nq-reader-to-wiki:
	python densephrases/scripts/create_nq_reader_wiki.py \
		$(DPH_DATA_DIR)/single-qa/nq/train.json,$(DPH_DATA_DIR)/single-qa/nq/dev.json \
		$(DPH_DATA_DIR)/single-qa/nq \
		$(DPH_DATA_DIR)/wikidump/20181220_concat/

download-wiki: data-config
	python densephrases/scripts/download_wikidump.py \
		--output_dir $(WIKI_DIR)

nq-reader-train: data-config
	python densephrases/scripts/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_path $(NQREADER_DIR)/train_79168.json

nq-reader-dev: data-config
	python densephrases/scripts/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_8757.json

nq-reader-dev-sample: data-config
	python densephrases/scripts/create_nq_reader.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_path $(NQREADER_DIR)/dev_sample.json

nq-reader-train-docs: data-config
	python densephrases/scripts/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/train.json \
		--output_dir $(NQREADER_DOC_DIR)/train

nq-reader-dev-docs: data-config
	python densephrases/scripts/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/train/nq-train-*.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)/dev

nq-reader-dev-docs-sample: data-config
	python densephrases/scripts/create_nq_reader_doc.py \
		--nq_orig_path_pattern "$(NQORIG_DIR)/sample/nq-train-sample.jsonl.gz" \
		--nq_open_path $(NQOPEN_DIR)/dev.json \
		--output_dir $(NQREADER_DOC_DIR)-sample

nq-_reader-train-docs-wiki: data-config
	python densephrases/scripts/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/train \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki

nq-reader-dev-docs-wiki: data-config
	python densephrases/scripts/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(NQREADER_DOC_DIR)/dev \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki

squad-reader-train-docs-wiki: data-config
	python densephrases/scripts/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/train-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/train_wiki \
		--is_squad

squad-reader-dev-docs-wiki: data-config
	python densephrases/scripts/create_nq_reader_doc_wiki.py \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--nq_reader_docs_dir $(SQUAD_DIR)/dev-v1.1.json \
		--output_dir $(SQUADREADER_DOC_DIR)/dev_wiki \
		--is_squad

build-db: data-config
	python densephrases/scripts/build_db.py \
		--data_path $(WIKI_DIR)/extracted \
		--save_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--preprocess densephrases/scripts/prep_wikipedia.py \
		--overwrite

build-wikisquad: data-config
	python densephrases/scripts/build_wikisquad.py \
		--db_path $(WIKI_DIR)/docs_20181220_nolist.db \
		--out_dir $(WIKI_DIR)/20181220_nolist

concat-wikisquad: data-config
	python densephrases/scripts/concat_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_concat

first-para-wikisquad: data-config
	python densephrases/scripts/first_para_wikisquad.py \
		--input_dir $(WIKI_DIR)/20181220_nolist \
		--output_dir $(WIKI_DIR)/20181220_nolist_first

compare-db: data-config
	python densephrases/scripts/compare_db.py \
		--db1 $(DPH_DATA_DIR)/denspi/docs.db \
		--db2 $(WIKI_DIR)/docs_20181220.db
		
sample-nq-reader-doc-wiki-train: data-config
	python densephrases/scripts/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.15 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/train_wiki \
		--output_dir $(NQREADER_DOC_DIR)/train_wiki_noise

sample-nq-reader-doc-wiki-dev: data-config
	python densephrases/scripts/sample_nq_reader_doc_wiki.py \
		--sampling_ratio 0.1 \
		--wiki_dir $(WIKI_DIR)/20181220_concat \
		--docs_wiki_dir $(NQREADER_DOC_DIR)/dev_wiki \
		--output_dir $(NQREADER_DOC_DIR)/dev_wiki_noise
