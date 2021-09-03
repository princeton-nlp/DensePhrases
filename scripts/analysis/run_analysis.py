# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import json
import argparse
import logging
import os
import copy
import torch
import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from densephrases.utils.squad_utils import get_cq_dataloader, SquadResult
from densephrases.utils.single_utils import set_seed, to_list, to_numpy, backward_compat
from densephrases.utils.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from densephrases.utils.embed_utils import write_phrases
from densephrases.models import DensePhrases


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)
print(set([k.split('-')[0] for k in ALL_MODELS]))


def dump_phrases(args, model, tokenizer):

    logger.info(f"Pair file from {args.pair_file}")
    passage_pairs = json.load(open(args.pair_file))['data']
    
    questions = [pp['question'] for pp in passage_pairs]
    answers = [pp['answer'][0] for pp in passage_pairs]
    answers_start = [pp['answer'][1] for pp in passage_pairs]
    titles = [pp['title'] for pp in passage_pairs]
    gold_passages = [pp['gold_passage'] for pp in passage_pairs]
    entail_neg_passages = [pp['entail_neg_passage'] for pp in passage_pairs]
    neg_titles = [pp['neg_title'] for pp in passage_pairs]
    topic_neg_passages = [pp['topic_neg_passage'] for pp in passage_pairs]
    
    def get_phrase_results(dataloader, eval_features):
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy()

        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        for batch in tqdm(dataloader, desc="Evaluating", disable=True):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "input_ids_": batch[8],
                    "attention_mask_": batch[9],
                    "token_type_ids_": batch[10],
                }
                feature_indices = batch[-1]
                outputs = model(**inputs)
            
            for i, feature_index in enumerate(feature_indices):
                eval_feature = eval_features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs[1:]]
                start_logits, end_logits, sft_logits, eft_logits = output
                result = SquadResult(
                    unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    sft_logits=sft_logits,
                    eft_logits=eft_logits,
                )
                yield result
    
    logger.info(f"***** Processing gold passages *****")
    
    # Gold passages
    dataloader, examples, features = get_cq_dataloader(
        gold_passages, questions, tokenizer, args.max_query_length, batch_size=64,
        answers=answers, answers_start=answers_start, titles=titles,
    )
    id2feature = {feature.unique_id: feature for feature in features}
    id2exid = {id_: id2feature[id_].example_index for id_ in id2feature}
    
    stats = {}
    # TODO: add filter + mask padding
    for feature_idx, result in tqdm(enumerate(get_phrase_results(dataloader, features))):
        if features[feature_idx].start_position <= 0:
            continue
        if id2exid[result.unique_id] not in stats:
            stats[id2exid[result.unique_id]] = {'gold_score': -1e9, 'topic_neg_score': -1e9, 'entail_neg_score': -1e9}

        gold_score = np.expand_dims(np.array(result.start_logits), 1) + np.expand_dims(np.array(result.end_logits), 0)
        gold_score[0,0] = 0

        filter_score = np.expand_dims(np.array(result.sft_logits) > args.filter_threshold, 1) * np.expand_dims(np.array(result.eft_logits) > args.filter_threshold, 0)
        maxlen = len(features[feature_idx].tokens) - 1
        filter_score[maxlen:, maxlen:] = 0
        gold_score = gold_score * filter_score

        gold_score = (np.triu(gold_score, 0) - np.triu(gold_score, 10)).max()
        stats[id2exid[result.unique_id]]['gold_score'] = (
            gold_score if gold_score > stats[id2exid[result.unique_id]]['gold_score'] else stats[id2exid[result.unique_id]]['gold_score']
        )
        # break

    logger.info(f"***** Processing topic neg passages *****")

    # Topic neg passages
    dataloader, examples, features = get_cq_dataloader(
        topic_neg_passages, questions, tokenizer, args.max_query_length, batch_size=64,
        answers=None, answers_start=None, titles=neg_titles,
    )
    id2feature = {feature.unique_id: feature for feature in features}
    id2exid = {id_: id2feature[id_].example_index for id_ in id2feature}
    assert all(example.start_position == -1 for example in examples)

    for feature_idx, result in tqdm(enumerate(get_phrase_results(dataloader, features))):
        if id2exid[result.unique_id] not in stats:
            continue

        topic_max_score = np.expand_dims(np.array(result.start_logits), 1) + np.expand_dims(np.array(result.end_logits), 0)
        topic_max_score[0,0] = 0

        filter_score = np.expand_dims(np.array(result.sft_logits) > args.filter_threshold, 1) * np.expand_dims(np.array(result.eft_logits) > args.filter_threshold, 0)
        maxlen = len(features[feature_idx].tokens) - 1
        filter_score[maxlen:, maxlen:] = 0
        topic_max_score = topic_max_score * filter_score

        topic_max_score = (np.triu(topic_max_score, 0) - np.triu(topic_max_score, 10)).max()
        stats[id2exid[result.unique_id]]['topic_neg_score'] = (
            topic_max_score if topic_max_score > stats[id2exid[result.unique_id]]['topic_neg_score'] else stats[id2exid[result.unique_id]]['topic_neg_score']
        )
        # break

    logger.info(f"***** Processing entail neg passages *****")

    # Entail neg passages
    dataloader, examples, features = get_cq_dataloader(
        entail_neg_passages, questions, tokenizer, args.max_query_length, batch_size=64,
        answers=None, answers_start=None, titles=titles,
    )
    id2feature = {feature.unique_id: feature for feature in features}
    id2exid = {id_: id2feature[id_].example_index for id_ in id2feature}
    assert all(example.start_position == -1 for example in examples)

    for feature_idx, result in tqdm(enumerate(get_phrase_results(dataloader, features))):
        if id2exid[result.unique_id] not in stats:
            print('id not found')
            continue

        entail_max_score = np.expand_dims(np.array(result.start_logits), 1) + np.expand_dims(np.array(result.end_logits), 0)
        entail_max_score[0,0] = 0

        filter_score = np.expand_dims(np.array(result.sft_logits) > args.filter_threshold, 1) * np.expand_dims(np.array(result.eft_logits) > args.filter_threshold, 0)
        maxlen = len(features[feature_idx].tokens) - 1
        filter_score[maxlen:, maxlen:] = 0
        entail_max_score = entail_max_score * filter_score

        entail_max_score = (np.triu(entail_max_score, 0) - np.triu(entail_max_score, 10)).max()
        stats[id2exid[result.unique_id]]['entail_neg_score'] = (
            entail_max_score if entail_max_score > stats[id2exid[result.unique_id]]['entail_neg_score'] else stats[id2exid[result.unique_id]]['entail_neg_score']
        )
        # break
    
    if not all(all(val > -999 for val in score.values()) for score in stats.values()):
        import pdb; pdb.set_trace()

    gold_mean = sum([stat['gold_score'] for stat in stats.values()]) / len(stats)
    topic_mean = sum([stat['topic_neg_score'] for stat in stats.values()]) / len(stats)
    entail_mean = sum([stat['entail_neg_score'] for stat in stats.values()]) / len(stats)

    L_topic = sum([
        -torch.nn.functional.log_softmax(torch.Tensor([stat['gold_score'], stat['topic_neg_score']]), dim=-1)[0]
        for stat in stats.values()
    ])/len(stats)
    L_hard = sum([
        -torch.nn.functional.log_softmax(torch.Tensor([stat['gold_score'], stat['entail_neg_score']]), dim=-1)[0]
        for stat in stats.values()
    ])/len(stats)

    logger.info(f'gold mean: {gold_mean:.2f}, topic mean: {topic_mean:.2f}, entail mean: {entail_mean:.2f}')
    logger.info(f'topical relevance: {gold_mean - topic_mean:.2f}, fine-grained entailment: {gold_mean - entail_mean:.2f}')
    logger.info(f'L_topic: {L_topic:.4f}, L_hard: {L_hard:.4f}')
    logger.info(f"Analysis done for {len(passage_pairs)} (processed={len(stats)})")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--pretrained_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        type=str,
        help="The load directory where the model checkpoints are saved. Set to output_dir if not specified.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--pair_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3 and pre-processed data",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--filter_threshold",
        type=float,
        default=-1e8,
        help="model-based filtering threshold.",
    )
    parser.add_argument(
        "--filter_threshold_list",
        type=str,
        default='-1e8',
        help="model-based filtering threshold for filter testing. comma seperated values are given",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--lambda_kl", default=0.0, type=float, help="Lambda for distillation")
    parser.add_argument("--lambda_neg", default=0.0, type=float, help="Lambda for in-batch negative")
    parser.add_argument("--lambda_flt", default=0.0, type=float, help="Lambda for filtering")
    parser.add_argument("--pbn_size", default=0, type=int, help="pre-batch negative size")
    parser.add_argument("--pbn_tolerance", default=9999, type=int, help="pre-batch tolerance epoch")
    parser.add_argument("--append_title", action="store_true", help="Whether to append title in context.")
    parser.add_argument("--do_analysis", action="store_true", help="Whether to run dumping on the dev set.")
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=12, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=12, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=10,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=9999999999, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--wandb", action="store_true", help="Whether to use Weights and Biases logging")
    parser.add_argument(
        "--draft", action="store_true", help="Run draft mode to use 20 examples only"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--dense_offset', default=-2, type=float) # Originally -2
    parser.add_argument('--dense_scale', default=20, type=float) # Originally 20
    parser.add_argument("--threads", type=int, default=20, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    logger.setLevel(logging.WARNING)
    args.model_type = args.model_type.lower()
    config, unused_kwargs = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        output_hidden_states=False,
        return_unused_kwargs=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Load pre-trained DensePhrases
    model = DensePhrases(
        config=config,
        tokenizer=tokenizer,
        pretrained=None,
        transformer_cls=MODEL_MAPPING[config.__class__],
        lambda_kl=args.lambda_kl,
        lambda_neg=args.lambda_neg,
        lambda_flt=args.lambda_flt,
        pbn_size=args.pbn_size,
    )
    logger.setLevel(logging.INFO)
    logger.info('Number of model params: {:,}'.format(sum(p.numel() for p in model.parameters())))


    # Dump phrases
    if args.do_analysis:
        assert args.load_dir
        args.draft = False

        # Load only phrase encoder
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
        ))
        model.to(args.device)
        logger.info(f'DensePhrases loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')
        logger.info('Number of model params while dumping: {:,}'.format(sum(p.numel() for p in model.parameters())))
        dump_phrases(args, model, tokenizer)


if __name__ == "__main__":
    main()
