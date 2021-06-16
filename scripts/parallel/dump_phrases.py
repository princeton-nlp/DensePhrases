import argparse
import math
import os
import subprocess


def run_dump_phrase(args):
    do_lower_case = '--do_lower_case' if args.do_lower_case else ''
    append_title = '--append_title' if args.append_title else ''
    def get_cmd(start_doc, end_doc):
        return ["python", "generate_phrase_vecs.py",
                "--model_type", f"{args.model_type}",
                "--pretrained_name_or_path", f"{args.pretrained_name_or_path}",
                "--data_dir", f"{args.phrase_data_dir}",
                "--cache_dir", f"{args.cache_dir}",
                "--predict_file", f"{start_doc}:{end_doc}",
                "--do_dump",
                "--max_seq_length", "512",
                "--doc_stride", "500",
                "--fp16",
                "--load_dir", f"{args.load_dir}",
                "--output_dir", f"{args.output_dir}",
                "--filter_threshold", f"{args.filter_threshold:.2f}"] + \
                ([f"{do_lower_case}"] if len(do_lower_case) > 0 else []) + \
                ([f"{append_title}"] if len(append_title) > 0 else [])

    num_docs = args.end - args.start
    num_gpus = args.num_gpus
    num_docs_per_gpu = int(math.ceil(num_docs / num_gpus))
    start_docs = list(range(args.start, args.end, num_docs_per_gpu))
    end_docs = start_docs[1:] + [args.end]

    print(start_docs)
    print(end_docs)

    for device_idx, (start_doc, end_doc) in enumerate(zip(start_docs, end_docs)):
        print(get_cmd(start_doc, end_doc))
        subprocess.Popen(get_cmd(start_doc, end_doc))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bert')
    parser.add_argument('--pretrained_name_or_path', default='SpanBERT/spanbert-base-cased')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--cache_dir', default='')
    parser.add_argument('--data_name', default='') # for suffix
    parser.add_argument('--load_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--do_lower_case', default=False, action='store_true')
    parser.add_argument('--append_title', default=False, action='store_true')
    parser.add_argument('--filter_threshold', default=-1e9, type=float)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=8, type=int)
    args = parser.parse_args()

    args.output_dir = args.output_dir + '_%s' % (os.path.basename(args.data_name))
    args.phrase_data_dir = os.path.join(args.data_dir, args.data_name)

    return args


def main():
    args = get_args()
    run_dump_phrase(args)


if __name__ == '__main__':
    main()
