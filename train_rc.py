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


import argparse
import logging
import os
import timeit
import copy
import h5py
import torch
import wandb

from time import time
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from densephrases.utils.squad_utils import SquadResult, load_and_cache_examples
from densephrases.utils.single_utils import set_seed, to_list, to_numpy, backward_compat
from densephrases.utils.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate
from densephrases.encoder import DensePhrases


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)
print(set([k.split('-')[0] for k in ALL_MODELS]))


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) # don't use DistributedSampler due to OOM. Will manually split dataset later.
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    def is_train_param(name, verbose=False):
        if name.endswith(".embeddings.word_embeddings.weight"):
            if verbose:
                logger.info(f'freezing {name}')
            return False
        if name.startswith("cross_encoder"):
            return False

        return True

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and is_train_param(n)
            ],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    logger.info('Number of trainable params: {:,}'.format(
        sum(p.numel() for n, p in model.named_parameters() if is_train_param(n, verbose=False)))
    )

    # Check if saved optimizer or scheduler states exist
    if args.load_dir:
        if os.path.isfile(os.path.join(args.load_dir, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.load_dir, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.load_dir, "optimizer.pt"), map_location=torch.device('cpu'))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.load_dir, "scheduler.pt"), map_location=torch.device('cpu'))
            )
            logger.info(f'optimizer and scheduler loaded from {args.load_dir}')

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.load_dir:
        try:
            # set global_step to global_step of last saved checkpoint from model path
            checkpoint_suffix = args.load_dir.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for ep_idx, _ in enumerate(train_iterator):
        logger.info(f"\n[Epoch {ep_idx+1}]")
        if args.pbn_size > 0 and ep_idx + 1 > args.pbn_tolerance:
            if hasattr(model, 'module'):
                model.module.init_pre_batch(args.pbn_size)
            else:
                model.init_pre_batch(args.pbn_size)
            logger.info(f"Initialize pre-batch of size {args.pbn_size} for Epoch {ep_idx+1}")

        # Skip batch
        train_iterator = iter(train_dataloader)
        initial_step = steps_trained_in_current_epoch
        if steps_trained_in_current_epoch > 0:
            train_dataloader.dataset.skip = True # used for LazyDataloader
            while steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                next(train_iterator)
            train_dataloader.dataset.skip = False
        assert steps_trained_in_current_epoch == 0

        epoch_iterator = tqdm(
            train_iterator, initial=initial_step, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        # logger.setLevel(logging.DEBUG)
        start_time = time()
        for step, batch in enumerate(epoch_iterator):
            logger.debug(f'1) {time()-start_time:.3f}s: on-the-fly pre-processing')
            start_time = time()

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "input_ids_": batch[8],
                "attention_mask_": batch[9],
                "token_type_ids_": batch[10],
                # "neg_input_ids": batch[12], # should be commented out if none
                # "neg_attention_mask": batch[13],
                # "neg_token_type_ids": batch[14],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            epoch_iterator.set_description(f"Loss={loss.item():.3f}, lr={scheduler.get_lr()[0]:.6f}")

            logger.debug(f'2) {time()-start_time:.3f}s: forward')
            start_time = time()

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logger.debug(f'3) {time()-start_time:.3f}s: backward')
            start_time = time()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                logger.debug(f'4) {time()-start_time:.3f}s: optimize')
                start_time = time()

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Validation acc
                        logger.setLevel(logging.WARNING)
                        results, _ = evaluate(args, model, tokenizer, prefix=global_step)
                        wandb.log(
                            {"Eval EM": results['exact'], "Eval F1": results['f1']}, step=global_step,
                        )
                        logger.setLevel(logging.INFO)

                    wandb.log(
                        {"lr": scheduler.get_lr()[0], "loss": (tr_loss - logging_loss) / args.logging_steps},
                        step=global_step
                    )
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = timeit.default_timer()

    def get_results():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "input_ids_": batch[6],
                    "attention_mask_": batch[7],
                    "token_type_ids_": batch[8],
                }
                feature_indices = batch[3]
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                # TODO: i and feature_index are the same number! Simplify by removing enumerate?
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                if len(output) != 4:
                    raise NotImplementedError
                else:
                    start_logits, end_logits, sft_logits, eft_logits = output
                    result = SquadResult(
                        unique_id,
                        start_logits=start_logits,
                        end_logits=end_logits,
                        sft_logits=sft_logits,
                        eft_logits=eft_logits,
                    )
                yield result

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    output_candidates_file = os.path.join(args.output_dir, "candidates_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        raise NotImplementedError
    else:
        predictions, stat = compute_predictions_logits(
            examples,
            features,
            get_results(),
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
            args.filter_threshold,
            output_candidates_file,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions, null_log_odds_file=output_null_log_odds_file)

    # Log stat locally
    with open('eval_logger.txt', 'a') as f:
        f.write(f'{args.output_dir}\t{results["exact"]:.3f}\t{results["f1"]:.3f}\n')

    evalTime = timeit.default_timer() - start_time
    logger.info("Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    return results, stat


def filter_test(args, model, tokenizer):
    original_filter_threshold = args.filter_threshold
    thresholds = [float(th) for th in args.filter_threshold_list.split(',')]
    logger.info(f'Testing following filters: {thresholds}')

    results = {}
    for idx, threshold in enumerate(thresholds):
        logger.info(f"Filter={threshold:.2f}")
        args.filter_threshold = threshold
        result, stat = evaluate(args, model, tokenizer, prefix=str(threshold))
        result = dict(
            (k + ("_{:.2f}".format(threshold)), v) for k, v in result.items() if k.startswith('exact') or k.startswith('f1')
        )
        result[f'save_rate_{threshold:.2f}'] = stat["save_rate"]
        results.update(result)

    logger.info("Results: {}".format(results))
    results = list(results.items())
    print('threshold\texact\tf1\tsave_rate')
    for idx in range(0, len(results), 3):
        print(f"{thresholds[idx//3]:.1f}, {results[idx][1]:.2f}, {results[idx+1][1]:.2f}, {results[idx+2][1]:.3f}")
    args.filter_threshold = original_filter_threshold


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
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--teacher_dir",
        default=None,
        type=str,
        help="The teacher directory where the model checkpoints are saved.",
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
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
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
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_filter_test", action="store_true", help="Whether to test filters.")
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)
    parser.add_argument("--truecase", action="store_true", help="Dummy (automatic truecasing supported)")
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
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
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

    parser.add_argument("--threads", type=int, default=20, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.local_rank != -1:
        if args.local_rank > 0:
            logger.setLevel(logging.WARN)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if args.draft:
        args.overwrite_output_dir = True
        args.logging_steps = 999999999 # Do not log
        logger.warning(f'Save model in {args.output_dir} in draft version')

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
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
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Set wandb
    if args.do_train or args.do_eval:
        wandb.init(project="DensePhrases (single)", mode="online" if args.wandb else "disabled")
        wandb.config.update(args)

    # Load config, tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

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

    # Load pre-trained LM
    pretrained = None
    if not args.load_dir:
        pretrained = AutoModel.from_pretrained(
            args.pretrained_name_or_path,
            from_tf=bool(".ckpt" in args.pretrained_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        logger.info(f'DensePhrases initialized with {pretrained.__class__}')
    model = DensePhrases(
        config=config,
        tokenizer=tokenizer,
        pretrained=copy.deepcopy(pretrained) if pretrained is not None else None,
        transformer_cls=MODEL_MAPPING[config.__class__],
        lambda_kl=args.lambda_kl,
        lambda_neg=args.lambda_neg,
        lambda_flt=args.lambda_flt,
    )
    logger.info('Number of model params: {:,}'.format(sum(p.numel() for p in model.parameters())))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"`
    # will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    global_step = 1
    tr_loss = 99999

    if args.do_train:
        # Load pre-trained DensePhrases
        if args.load_dir:
            model_dict = torch.load(os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
            model_dict = {
                n: p for n, p in model_dict.items() if not (n.startswith('cross_encoder') or n.startswith('qa_outputs'))
            }
            assert not any(param.startswith('cross_encoder') for param in model_dict.keys())
            model.load_state_dict(model_dict)
            logger.info(f'DensePhrases loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')

        # Load pre-trained cross encoder
        if args.lambda_kl > 0 and args.do_train:
            cross_encoder = torch.load(
                os.path.join(args.teacher_dir, "pytorch_model.bin"), map_location=torch.device('cpu')
            )
            new_qd = {n[len('bert')+1:]: p for n, p in cross_encoder.items() if 'bert' in n}
            new_linear = {n[len('qa_outputs')+1:]: p for n, p in cross_encoder.items() if 'qa_outputs' in n}
            qd_config, unused_kwargs = AutoConfig.from_pretrained(
                args.pretrained_name_or_path,
                cache_dir=args.cache_dir if args.cache_dir else None,
                return_unused_kwargs=True
            )
            qd_pretrained = AutoModel.from_pretrained(
                args.pretrained_name_or_path,
                from_tf=bool(".ckpt" in args.pretrained_name_or_path),
                config=qd_config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.cross_encoder = qd_pretrained
            model.cross_encoder.load_state_dict(new_qd)
            model.qa_outputs = torch.nn.Linear(config.hidden_size, 2)
            model.qa_outputs.load_state_dict(new_linear)
            logger.info(f'Distill with teacher model {args.teacher_dir}')

        # Train model
        model.to(args.device)
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False, skip_no_answer=False
        )
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if (args.do_train) and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Remove teacher before saving
        if args.lambda_kl > 0:
            del model.cross_encoder
            del model.qa_outputs

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model

        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
        ))
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

        # Set load_dir to trained model
        args.load_dir = args.output_dir
        logger.info(f'Will load {args.load_dir} that was trained.')

    # Test filter
    if args.do_filter_test:
        assert args.load_dir
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
        ))
        model.to(args.device)
        logger.info(f'DensePhrases loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')
        filter_test(args, model, tokenizer)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        assert args.load_dir
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
        ))
        model.to(args.device)
        logger.info(f'DensePhrases loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')
        result, _ = evaluate(args, model, tokenizer, prefix='final')
        result = dict((k + "_final", v) for k, v in result.items())
        wandb.log(
                {"Eval EM": result['exact_final'], "Eval F1": result['f1_final'], "loss": tr_loss}, step=global_step,
        )
        logger.info("Results: {}".format(result))


if __name__ == "__main__":
    main()
