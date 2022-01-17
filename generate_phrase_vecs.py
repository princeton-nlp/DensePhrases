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


import logging
import os
import timeit
import h5py

from transformers.data.data_collator import default_data_collator, torch_default_data_collator
from datasets import load_dataset


from densephrases.utils.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    DataCollatorWithPadding,
    default_data_collator,
)
from densephrases.utils.single_utils import set_seed, load_encoder
from densephrases import Options
from scripts.preprocess.convert_squad_to_hf import convert_squad_to_hf

logger = logging.getLogger(__name__)


def dump_phrases(args, model, tokenizer, filter_only=False):
    output_path = 'dump/phrase' if not filter_only else 'dump/filter'
    if not os.path.exists(os.path.join(args.output_dir, output_path)):
        os.makedirs(os.path.join(args.output_dir, output_path))

    start_time = timeit.default_timer()
    if ':' not in args.test_file:
        test_files = [args.test_file]
        offsets = [0]
        output_dump_file = os.path.join(
            args.output_dir, f"{output_path}/{os.path.splitext(os.path.basename(args.test_file))[0]}.hdf5"
        )
    else:
        dirname = os.path.dirname(args.test_file)
        basename = os.path.basename(args.test_file)
        start, end = list(map(int, basename.split(':')))
        output_dump_file = os.path.join(
            args.output_dir, f"{output_path}/{start}-{end}.hdf5"
        )

        # skip files if possible
        if os.path.exists(output_dump_file):
            with h5py.File(output_dump_file, 'r') as f:
                dids = list(map(int, f.keys()))
            start = int(max(dids) / 1000)
            logger.info('%s exists; starting from %d' % (output_dump_file, start))

        names = [str(i).zfill(4) for i in range(start, end)]
        test_files = [os.path.join(dirname, name) for name in names]
        offsets = [int(each) * 1000 for each in names]

    for offset, test_file in zip(offsets, test_files):
        logger.info(f"***** Pre-processing contexts from {test_file} *****")

        data_files = {}
        if args.convert_squad_to_hf:
            data_files["test"] = convert_squad_to_hf(test_file)
        else:
            data_files["test"] = test_file
        extension = data_files["test"].split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=args.cache_dir)

        column_names = raw_datasets["test"].column_names
        context_column_name = "context" if "context" in column_names else column_names[1]

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

        if args.max_seq_length > tokenizer.model_max_length:
            args.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

        # Validation preprocessing
        def prepare_validation_features(examples, indexes):
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

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
            
            # Inflate doc_idxs based on sample_mapping
            tokenized_examples['doc_idx'] = [offset + examples['doc_idx'][i] for i in sample_mapping]

            # This example_id indicates the index of an original paragraph (not question id)
            tokenized_examples['example_id'] = [indexes[i] for i in sample_mapping]

            for i in range(len(tokenized_examples["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1 if pad_on_right and args.append_title else 0

                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples
        
        examples = raw_datasets["test"]
        
        # Predict Feature Creation
        with args.main_process_first(desc="prediction dataset map pre-processing"):
            dataset = examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                # num_proc=1,
                with_indices=True,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        
        # Data collator
        # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
        # collator.
        data_collator = (
            default_data_collator
            if args.pad_to_max_length
            else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
        )

        # Use trainer for predict
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.generate_phrase_vecs(dataset, examples, output_dump_file, offset, args)

        evalTime = timeit.default_timer() - start_time
        logger.info("Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))


def main():
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_data_options()
    options.add_rc_options()
    args = options.parse()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    logger.info("Dump parameters %s", args)

    # TODO: FP16 for generate_phrase_vecs
    assert args.load_dir
    model, tokenizer, _ = load_encoder(args.device, args, phrase_only=True)

    # Create phrase vectors
    args.draft = False
    dump_phrases(args, model, tokenizer, filter_only=args.filter_only)


if __name__ == "__main__":
    main()
