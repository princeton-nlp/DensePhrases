import json
import os
import argparse

from tqdm import tqdm


def filter_wiki(args):
    if not os.path.exists(args.to_dir):
        os.makedirs(args.to_dir)

    names = os.listdir(args.from_dir)
    from_paths = [os.path.join(args.from_dir, name) for name in names]
    to_paths = [os.path.join(args.to_dir, name) for name in names]

    for from_path, to_path in zip(tqdm(from_paths), to_paths):
        with open(from_path, 'r') as fp:
            from_ = json.load(fp)
        to = {'data': []}
        for article in from_['data']:
            to_article = {'paragraphs': [], 'title': article['title']}
            for para in article['paragraphs']:
                if args.min_num_chars <= len(para['context']) < args.max_num_chars:
                    to_article['paragraphs'].append(para)
            to['data'].append(to_article)

        with open(to_path, 'w') as fp:
            json.dump(to, fp)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('from_dir')
    parser.add_argument('to_dir')
    parser.add_argument('--min_num_chars', default=250, type=int)
    parser.add_argument('--max_num_chars', default=2500, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    filter_wiki(args)


if __name__ == '__main__':
    main()
