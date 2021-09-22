# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_model_options(self):
        self.parser.add_argument("--model_type", type=str, default='bert',
                        help="Model type selected in the list",)
        self.parser.add_argument("--pretrained_name_or_path", type=str, default='SpanBERT/spanbert-base-cased',
                        help="Path to pre-trained model or shortcut name selected in the list",)
        self.parser.add_argument("--config_name", type=str, default="",
                        help="Pretrained config name or path if not the same as model_name")
        self.parser.add_argument("--tokenizer_name", type=str, default="",
                        help="Pretrained tokenizer name or path if not the same as model_name",)
        self.parser.add_argument("--load_dir", type=str, default="",
                        help="load dir where the model checkpoints are saved. Set to output_dir if not specified.",)
        self.parser.add_argument("--output_dir", type=str, default=None,
                        help="The output directory where the model checkpoints and predictions will be written.",)
        self.parser.add_argument("--max_seq_length", type=int, default=384,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                        "longer than this will be truncated, and sequences shorter than this will be padded.",)
        self.parser.add_argument("--doc_stride", type=int, default=128,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.",)
        self.parser.add_argument("--max_query_length", type=int, default=64,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                        "be truncated to this length.",)
        self.parser.add_argument("--max_answer_length", type=int, default=10,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                        "and end predictions are not conditioned on one another.",)
        self.parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    def add_index_options(self):
        self.parser.add_argument('--stage', type=str)
        self.parser.add_argument('--dump_dir', type=str)
        self.parser.add_argument('--offset', type=int, default=0)
        self.parser.add_argument('--quantizer_path', default='quantizer.faiss')
        self.parser.add_argument('--trained_index_path', default='trained.faiss')
        self.parser.add_argument('--phrase_dir', default='phrase')
        self.parser.add_argument('--index_name', default='start/256_flat_SQ4')
        self.parser.add_argument('--index_path', default='index.faiss')
        self.parser.add_argument('--idx2id_path', default='idx2id.hdf5')
        self.parser.add_argument('--add_all', action='store_true', default=False)
        self.parser.add_argument('--num_clusters', type=int, default=16384)
        self.parser.add_argument('--hnsw', action='store_true', default=False)
        self.parser.add_argument('--fine_quant', default='SQ4', help='SQ4|OPQ# where # is number of bytes per vector')
        self.parser.add_argument('--norm_th', type=float, default=999)
        self.parser.add_argument('--doc_sample_ratio', type=float, default=0.2)
        self.parser.add_argument('--vec_sample_ratio', type=float, default=0.2)
        self.parser.add_argument('--cuda', action='store_true', default=False)
        self.parser.add_argument('--replace', action='store_true', default=False)
        self.parser.add_argument('--num_docs_per_add', type=int, default=2000)
        self.parser.add_argument('--first_passage', action='store_true', default=False, help="dump only first passages")
        self.parser.add_argument("--index_filter", type=float, default=-1e8, help="filtering threshold for index",)

        # For merging IVFSQ subindexes
        self.parser.add_argument('--dump_paths', default=None,
                        help='Relative to `dump_dir/phrase`. If specified, creates subindex dir')
        self.parser.add_argument('--inv_path', default='merged.invdata')
        self.parser.add_argument('--subindex_name', default='index', help='used only if dump_path is specified.')

    def add_data_options(self):
        self.parser.add_argument("--data_dir", type=str, default=None,
                        help="The input data dir. Should contain the .json files for the task.",)
        self.parser.add_argument("--cache_dir", type=str, default="",
                        help="Where do you want to store the pre-trained models and data downloaded from s3",)
        self.parser.add_argument("--threads", type=int, default=20,
                        help="multiple threads for converting example to features")
        self.parser.add_argument("--truecase_path", type=str, default='truecase/english_with_questions.dist')
        self.parser.add_argument("--truecase", action="store_true", help="Dummy (automatic truecasing supported)")

    # Reading comprehension (single-passage training) options
    def add_rc_options(self):
        self.parser.add_argument("--teacher_dir", type=str, default=None,
                        help="The teacher directory where the model checkpoints are saved.",)
        self.parser.add_argument("--train_file", type=str, default=None,
                        help="The input training file. If a data dir is specified, will look for the file there",)
        self.parser.add_argument("--predict_file", type=str, default=None,
                        help="The input evaluation file. If a data dir is specified, will look for the file there",)
        self.parser.add_argument("--version_2_with_negative", action="store_true",
                        help="If true, the SQuAD examples contain some that do not have an answer.",)
        self.parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.",)
        self.parser.add_argument("--filter_threshold_list", type=str, default='-1e8',
                        help="model-based filtering threshold for filter testing. comma seperated values are given",)
        self.parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        self.parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        self.parser.add_argument("--do_filter_test", action="store_true", help="Whether to test filters.")
        self.parser.add_argument("--lambda_kl", default=0.0, type=float, help="Lambda for distillation")
        self.parser.add_argument("--lambda_neg", default=0.0, type=float, help="Lambda for in-batch negative")
        self.parser.add_argument("--lambda_flt", default=0.0, type=float, help="Lambda for filtering")
        self.parser.add_argument("--pbn_size", default=0, type=int, help="pre-batch negative size")
        self.parser.add_argument("--pbn_tolerance", default=9999, type=int, help="pre-batch tolerance epoch")
        self.parser.add_argument("--append_title", action="store_true", help="Whether to append title in context.")
        self.parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Run evaluation during training at each logging step.")
        self.parser.add_argument("--per_gpu_train_batch_size", type=int, default=12,
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", type=int, default=12,
                        help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
        self.parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
        self.parser.add_argument("--num_train_epochs", type=float, default=2.0,
                        help="Total number of training epochs to perform.")
        self.parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
        self.parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        self.parser.add_argument("--n_best_size", type=int, default=20,
                        help="The total number of n-best predictions to generate in the json output file.",)
        self.parser.add_argument("--lang_id", type=int, default=0,
                        help="language id of input for language-specific xlm models",)
        self.parser.add_argument("--logging_steps", type=int, default=5000, help="Log every X updates steps.")
        self.parser.add_argument("--save_steps", type=int, default=9999999, help="Save checkpoint every X updates steps.")
        self.parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name",)
        self.parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
        self.parser.add_argument("--wandb", action="store_true", help="Whether to use Weights and Biases logging")
        self.parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
        self.parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

        # For generating phrase vectors
        self.parser.add_argument("--do_dump", action="store_true", help="Whether to generate phrase vecs")
        self.parser.add_argument("--filter_only", action="store_true", help="Whether to generate filter logits only")
        self.parser.add_argument('--dense_offset', type=float, default=-2)
        self.parser.add_argument('--dense_scale', type=float, default=20)
        self.parser.add_argument("--filter_threshold", type=float, default=-1e8, help="model-based filtering threshold.",)

    def add_retrieval_options(self):
        self.parser.add_argument('--run_mode', default='eval')
        self.parser.add_argument('--top_k', type=int, default=10)
        self.parser.add_argument('--nprobe', type=int, default=256)
        self.parser.add_argument('--aggregate', action='store_true', default=False)
        self.parser.add_argument('--agg_strat', type=str, default='opt1')

        # Evaluation
        self.parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
        self.parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
        self.parser.add_argument('--candidate_path', default=None)
        self.parser.add_argument('--regex', action='store_true', default=False)
        self.parser.add_argument('--eval_batch_size', type=int, default=64)
        self.parser.add_argument('--save_pred', action='store_true', default=False)

        # Passage retrieval options
        self.parser.add_argument('--eval_psg', action='store_true', default=False)
        self.parser.add_argument('--psg_top_k', type=int, default=100)
        self.parser.add_argument('--max_psg_len', type=int, default=999999999, help='used for fair comparison')
        self.parser.add_argument('--mark_phrase', action='store_true', default=False)
        self.parser.add_argument('--return_sent', action='store_true', default=False)
        self.parser.add_argument('--sent_window', type=int, default=0)

        # KILT
        self.parser.add_argument('--is_kilt', action='store_true', default=False)
        self.parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
        self.parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')

    # Query-side fine-tuning options
    def add_qsft_options(self):
        self.parser.add_argument('--train_path', default=None)
        self.parser.add_argument('--per_gpu_train_batch_size', default=48, type=int)
        self.parser.add_argument('--num_train_epochs', default=10, type=float)
        self.parser.add_argument("--learning_rate", default=3e-5, type=float)
        self.parser.add_argument("--warmup_steps", default=0.1, type=int)
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        self.parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        self.parser.add_argument('--label_strat', default='phrase', type=str, help="label strat={phrase|doc|phrase,doc}")

    def add_demo_options(self):
        self.parser.add_argument('--base_ip', default='http://127.0.0.1')
        self.parser.add_argument("--query_port", type=str, default='-1')
        self.parser.add_argument('--index_port', type=str, default='-1')
        self.parser.add_argument('--examples_path', default='examples.txt')

    def initialize_parser(self):
        self.parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        self.parser.add_argument("--draft", action="store_true",
                        help="Run draft mode to use small number of examples only")
        self.parser.add_argument("--verbose_logging", action="store_true",
                        help="If true, all of the warnings related to data processing will be printed. "
                        "A number of warnings are expected for a normal SQuAD evaluation.",)
        self.parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
        self.parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html",)

    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()

        # Option sanity checks
        if hasattr(opt, 'doc_stride'):
            if opt.doc_stride >= opt.max_seq_length - opt.max_query_length:
                logger.warning(
                    "WARNING - You've set a doc stride which may be superior to the document length in some examples." 
                )

        # Is overwriting?
        if hasattr(opt, 'output_dir') and hasattr(opt, 'do_train'):
            if (os.path.exists(opt.output_dir) and os.listdir(opt.output_dir) and opt.do_train
                and not opt.overwrite_output_dir):
                raise ValueError(
                    "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir.".format(
                        opt.output_dir)
                )

        # Is draft?
        if opt.draft:
            opt.overwrite_output_dir = True
            opt.logging_steps = 999999999 # Do not log
            logger.warning(f'Overwrite model in {opt.output_dir} in draft version')

        return opt
