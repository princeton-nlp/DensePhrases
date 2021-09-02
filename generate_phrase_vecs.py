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
from densephrases.utils.single_utils import set_seed, to_list, to_numpy, backward_compat
from densephrases.utils.embed_utils import write_phrases
from densephrases.encoder import DensePhrases


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)
print(set([k.split('-')[0] for k in ALL_MODELS]))


def dump_phrases(args, model, tokenizer):
    if not os.path.exists(os.path.join(args.output_dir, 'dump/phrase')):
        os.makedirs(os.path.join(args.output_dir, 'dump/phrase'))

    start_time = timeit.default_timer()
    if ':' not in args.predict_file:
        predict_files = [args.predict_file]
        offsets = [0]
        output_dump_file = os.path.join(
            args.output_dir, "dump/phrase/{}.hdf5".format(os.path.splitext(os.path.basename(args.predict_file))[0])
        )
    else:
        dirname = os.path.dirname(args.predict_file)
        basename = os.path.basename(args.predict_file)
        start, end = list(map(int, basename.split(':')))
        output_dump_file = os.path.join(
            args.output_dir, f"dump/phrase/{start}-{end}.hdf5"
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

        write_phrases(
            examples, features, get_phrase_results(), args.max_answer_length, args.do_lower_case, tokenizer,
            output_dump_file, args.filter_threshold, args.verbose_logging,
            args.dense_offset, args.dense_scale, has_title=args.append_title,
        )

        evalTime = timeit.default_timer() - start_time
        logger.info("Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))


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

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
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
        "--filter_threshold",
        type=float,
        default=-1e8,
        help="model-based filtering threshold.",
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
    parser.add_argument("--append_title", action="store_true", help="Whether to append title in context.")
    parser.add_argument("--do_dump", action="store_true", help="Whether to run dumping on the dev set.")
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--max_answer_length",
        default=10,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=12, type=int, help="Batch size per GPU/CPU for evaluation."
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

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
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

    # Load pre-trained LM
    pretrained = None
    assert args.load_dir
    model = DensePhrases(
        config=config,
        tokenizer=tokenizer,
        transformer_cls=MODEL_MAPPING[config.__class__],
    )
    logger.info('Number of model params: {:,}'.format(sum(p.numel() for p in model.parameters())))

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
        args.draft = False

        # Load only phrase encoder
        model.load_state_dict(backward_compat(
            torch.load(os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
        ))
        if hasattr(model, "module"):
            del model.module.query_start_encoder
            del model.module.query_end_encoder
        else:
            del model.query_start_encoder
            del model.query_end_encoder
        model.to(args.device)
        logger.info(f'DensePhrases loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')
        logger.info('Number of model params while dumping: {:,}'.format(sum(p.numel() for p in model.parameters())))
        dump_phrases(args, model, tokenizer)


if __name__ == "__main__":
    main()
