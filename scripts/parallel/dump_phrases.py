import argparse
import math
import os
import subprocess


def run_dump_phrase(args):
    append_title = '--append_title' if args.append_title else ''
    def get_cmd(test_dir, start_doc, end_doc):
        return ["python", "generate_phrase_vecs.py",
                "--pretrained_name_or_path", f"{args.pretrained_name_or_path}",
                "--cache_dir", f"{args.cache_dir}",
                "--test_file", f"{test_dir}/{start_doc}:{end_doc}",
                "--do_dump",
                "--max_seq_length", "512",
                "--doc_stride", "462",
                "--fp16",
                "--filter_threshold", f"{args.filter_threshold:.2f}",
                "--load_dir", f"{args.load_dir}",
                "--output_dir", f"{args.output_dir}"] + \
                ([f"{append_title}"] if len(append_title) > 0 else [])

    num_docs = args.end - args.start
    num_docs_per_proc = int(math.ceil(num_docs / args.num_procs))
    start_docs = list(range(args.start, args.end, num_docs_per_proc))
    end_docs = start_docs[1:] + [args.end]

    print(args.test_dir)
    print(start_docs)
    print(end_docs)

    for device_idx, (start_doc, end_doc) in enumerate(zip(start_docs, end_docs)):
        cmd = get_cmd(args.test_dir, start_doc, end_doc)
        subprocess.Popen(cmd)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_name_or_path', default='SpanBERT/spanbert-base-cased')
    parser.add_argument('--cache_dir', default='')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--dump_name', default='') # for suffix
    parser.add_argument('--load_dir', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--filter_threshold', default=-1e9, type=float)
    parser.add_argument('--append_title', default=False, action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=8, type=int)
    parser.add_argument('--num_procs', default=1, type=int)
    args = parser.parse_args()

    args.output_dir = args.output_dir + '_%s' % (os.path.basename(args.dump_name))
    args.test_dir = os.path.join(args.test_dir, args.dump_name)

    return args


def main():
    args = get_args()
    run_dump_phrase(args)


if __name__ == '__main__':
    main()
