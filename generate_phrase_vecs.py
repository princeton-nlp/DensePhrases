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

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from densephrases.utils.squad_utils import ContextResult, load_and_cache_examples
from densephrases.utils.single_utils import set_seed, to_list, to_numpy, backward_compat, load_encoder
from densephrases.utils.embed_utils import write_phrases, write_filter
from densephrases import Options

logger = logging.getLogger(__name__)


def dump_phrases(args, model, tokenizer, filter_only=False):
    output_path = 'dump/phrase' if not filter_only else 'dump/filter'
    if not os.path.exists(os.path.join(args.output_dir, output_path)):
        os.makedirs(os.path.join(args.output_dir, output_path))

    start_time = timeit.default_timer()
    if ':' not in args.predict_file:
        predict_files = [args.predict_file]
        offsets = [0]
        output_dump_file = os.path.join(
            args.output_dir, f"{output_path}/{os.path.splitext(os.path.basename(args.predict_file))[0]}.hdf5"
        )
    else:
        dirname = os.path.dirname(args.predict_file)
        basename = os.path.basename(args.predict_file)
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
        predict_files = [os.path.join(dirname, name) for name in names]
        offsets = [int(each) * 1000 for each in names]

    for offset, predict_file in zip(offsets, predict_files):
        args.predict_file = predict_file
        logger.info(f"***** Pre-processing contexts from {args.predict_file} *****")
        dataset, examples, features = load_and_cache_examples(
            args, tokenizer, evaluate=True, output_examples=True, context_only=True
        )
        for example in examples:
            example.doc_idx += offset

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info(f"***** Dumping Phrases from {args.predict_file} *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        start_time = timeit.default_timer()

        def get_phrase_results():
            for batch in tqdm(eval_dataloader, desc="Dumping"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "return_phrase": True,
                    }
                    feature_indices = batch[3]
                    outputs = model(**inputs)

                for i, feature_index in enumerate(feature_indices):
                    # TODO: i and feature_index are the same number! Simplify by removing enumerate?
                    eval_feature = features[feature_index.item()]
                    unique_id = int(eval_feature.unique_id)

                    output = [
                        to_numpy(output[i]) if type(output) != dict else {k: to_numpy(v[i]) for k, v in output.items()}
                        for output in outputs
                    ]

                    if len(output) != 4:
                        raise NotImplementedError
                    else:
                        start_vecs, end_vecs, sft_logits, eft_logits = output
                        result = ContextResult(
                            unique_id,
                            start_vecs=start_vecs,
                            end_vecs=end_vecs,
                            sft_logits=sft_logits,
                            eft_logits=eft_logits,
                        )
                    yield result

        if not filter_only:
            write_phrases(
                examples, features, get_phrase_results(), args.max_answer_length, args.do_lower_case, tokenizer,
                output_dump_file, args.filter_threshold, args.verbose_logging,
                args.dense_offset, args.dense_scale, has_title=args.append_title,
            )
        else:
            write_filter(
                examples, features, get_phrase_results(), tokenizer,
                output_dump_file, args.filter_threshold, args.verbose_logging, has_title=args.append_title,
            )

        evalTime = timeit.default_timer() - start_time
        logger.info("Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))


def main():
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_data_options()
    options.add_rc_options()
    args = options.parse()

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

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info("Dump parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"`
    # will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Create phrase vectors
    if args.do_dump:
        assert args.load_dir
        model, tokenizer, config = load_encoder(device, args, phrase_only=True)

        args.draft = False
        dump_phrases(args, model, tokenizer, filter_only=args.filter_only)


if __name__ == "__main__":
    main()
