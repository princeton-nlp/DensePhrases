#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
import os
import sys
import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, IntervalStrategy
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from densephrases import Options
from densephrases.utils.utils_qa import postprocess_qa_predictions
from densephrases.utils.trainer_qa import QuestionAnsweringTrainer
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.data_utils import TrueCaser
from scripts.preprocess.convert_squad_to_hf import convert_squad_to_hf


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.13.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_data_options()
    options.add_rc_options()
    args = options.parse()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if args.train_file is not None:
        if args.convert_squad_to_hf:
            data_files["train"] = convert_squad_to_hf(args.train_file)
        else:
            data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]

    if args.validation_file is not None:
        if args.convert_squad_to_hf:
            data_files["validation"] = convert_squad_to_hf(args.validation_file)
        else:
            data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
    if args.test_file is not None:
        if args.convert_squad_to_hf:
            data_files["test"] = convert_squad_to_hf(args.test_file)
        else:
            data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Initialize or load encoder
    model, tokenizer, _ = load_encoder(args.device, args)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if args.do_train:
        column_names = raw_datasets["train"].column_names
    elif args.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        args.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    max_query_length = args.max_query_length

    logger.info('Loading truecaser')
    if args.truecase:
        truecase = TrueCaser(os.path.join(os.environ['DATA_DIR'], args.truecase_path))

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if args.append_title:
            tokenized_examples = tokenizer(
                examples['title' if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else 'title'],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if args.pad_to_max_length else False,
            )
        else:
            tokenized_examples = tokenizer(
                examples[context_column_name],
                truncation="only_first",
                max_length=max_seq_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if args.pad_to_max_length else False,
            )

        tokenized_questions = tokenizer(
            examples[question_column_name],
            truncation="only_first",
            max_length=max_query_length,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Inflate questions based on sample_mapping
        tokenized_examples['input_ids_'] = [tokenized_questions['input_ids'][i] for i in sample_mapping]
        tokenized_examples['token_type_ids_'] = [tokenized_questions['token_type_ids'][i] for i in sample_mapping]
        tokenized_examples['attention_mask_'] = [tokenized_questions['attention_mask'][i] for i in sample_mapping]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right and args.append_title else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_index:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_index:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.truecase:
            for idx in range(len(train_dataset)):
                q = train_dataset[idx][question_column_name]
                train_dataset[idx][question_column_name] = truecase.get_true_case(q) if q == q.lower() else q

        # Create train feature from dataset
        with args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                # num_proc=1,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        if args.append_title:
            tokenized_examples = tokenizer(
                examples['title' if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else 'title'],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if args.pad_to_max_length else False,
            )
        else:
            tokenized_examples = tokenizer(
                examples[context_column_name],
                truncation="only_first",
                max_length=max_seq_length,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if args.pad_to_max_length else False,
            )

        tokenized_questions = tokenizer(
            examples[question_column_name],
            truncation="only_first",
            max_length=max_query_length,
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # Inflate questions based on sample_mapping
        tokenized_examples['input_ids_'] = [tokenized_questions['input_ids'][i] for i in sample_mapping]
        tokenized_examples['token_type_ids_'] = [tokenized_questions['token_type_ids'][i] for i in sample_mapping]
        tokenized_examples['attention_mask_'] = [tokenized_questions['attention_mask'][i] for i in sample_mapping]

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right and args.append_title else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        if args.truecase:
            for idx in range(len(eval_examples)):
                q = eval_examples[idx][question_column_name]
                eval_examples[idx][question_column_name] = truecase.get_true_case(q) if q == q.lower() else q

        # Validation Feature Creation
        with args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        if args.truecase:
            for idx in range(len(predict_examples)):
                q = predict_examples[idx][question_column_name]
                predict_examples[idx][question_column_name] = truecase.get_true_case(q) if q == q.lower() else q

        # Predict Feature Creation
        with args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval", filter_threshold=-1e5):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, save_rate = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            log_level=log_level,
            prefix=stage,
            filter_threshold=filter_threshold,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references), save_rate

    metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    
    if args.evaluate_during_training:
        args.evaluation_strategy = IntervalStrategy("steps")
        args.save_steps = int(len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs) // 10
        args.eval_steps = args.save_steps
        args.load_best_model_at_end = True
        args.save_total_limit = 5
        args.metric_for_best_model = "eval_exact_match"
        args.greater_is_better = True
        logger.info(f"Will save the model for every {args.save_steps} steps")

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        eval_examples=eval_examples if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Remove teacher from the model before saving
        if args.lambda_kl > 0:
            del trainer.model.cross_encoder
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples if args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = args.max_eval_samples if args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            args.max_predict_samples if args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    # Filter test
    if args.do_filter_test:
        thresholds = [float(th) for th in args.filter_threshold_list.split(',')]
        logger.info(f'Testing following filter thresholds: {thresholds}')

        results = []
        for idx, threshold in enumerate(thresholds):
            logger.info(f"Filter={threshold:.2f}")
            metrics = trainer.evaluate(filter_threshold=threshold)
            metrics['threshold'] = threshold
            results.append(metrics)

        logger.info("Filter Results")
        for idx in range(len(results)):
            out_str = ''
            for key, val in results[idx].items():
                out_str += f'{key}: {val:.2f} '
            logger.info(out_str)

    kwargs = {"finetuned_from": args.pretrained_name_or_path, "tasks": "question-answering"}

    # if args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
