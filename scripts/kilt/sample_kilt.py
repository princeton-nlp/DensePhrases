import json
import argparse
import os
import random
import time
import numpy as np

from tqdm import tqdm


def main(input_file, num_sample, balanced):
    print('reading', input_file)
    random.seed(999)
    np.random.seed(999)

    examples = json.load(open(input_file))['data']
    print(f'sampling from {len(examples)}')
    relation_dict = {}
    for example in tqdm(examples):
        relation = example['question'].split(' [SEP] ')[-1]
        if relation not in relation_dict:
            relation_dict[relation] = []
        relation_dict[relation].append(example)

    top_relations = sorted(relation_dict.items(), key=lambda x: len(x[1]), reverse=True)
    print('There are', len(relation_dict), 'relations.')
    print([(rel, len(rel_list)) for rel, rel_list in top_relations])
    print()
    exit()

    if not balanced:
        sample_per_relation = {
            rel: int((len(rel_list)/len(examples)) * num_sample) + 1 for rel, rel_list in top_relations
        }
    else:
        sample_per_relation = {
            rel: min(num_sample, len(rel_list)) for rel, rel_list in top_relations
        }
    print('Sample following number of relations')
    print(sample_per_relation)

    sample_examples = []
    for rel, rel_list in relation_dict.items():
        sample_idx = np.random.choice(len(rel_list), size=(sample_per_relation[rel]), replace=False)
        sample_examples += np.array(rel_list)[sample_idx].tolist()

    out_file = input_file.replace('.json', f'_{num_sample}_{"balanced" if balanced else "ratio"}.json')
    print(f'Saving {len(sample_examples)} examples to {out_file}')
    with open(out_file, 'w') as f:
        json.dump({'data': sample_examples}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--num_sample", type=int, required=True)
    parser.add_argument("--balanced", action='store_true', default=False)
    
    args = parser.parse_args()
    
    main(args.input_file, args.num_sample, args.balanced)
