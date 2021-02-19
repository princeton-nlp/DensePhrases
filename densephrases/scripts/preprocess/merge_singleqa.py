import json
import argparse
import os

from tqdm import tqdm


def merge_single(input_dir, output_path):

    paths = [
        'single-qa/nq/train_wiki3.json',
        'single-qa/webq/webq-train_ds.json',
        'single-qa/trec/trec-train_ds.json',
        'single-qa/tqa/tqa-train_ds.json',
        # 'single-qa/squad/train-v1.1.json',
    ]
    paths = [os.path.join(input_dir, path) for path in paths]
    assert all([os.path.exists(path) for path in paths])

    data_to_save = []
    sep_cnt = 0
    for path in paths:
        with open(path) as f:
            data = json.load(f)['data']
            data_to_save += data
        print(f'{path} has {len(data)} PQA triples')

    print(f'Saving {len(data_to_save)} RC triples to output_path')
    print('Writing to %s\n'% output_path)
    with open(output_path, 'w') as f:
        json.dump({'data': data_to_save}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, default=None)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    merge_single(args.input_dir, args.output_path)
