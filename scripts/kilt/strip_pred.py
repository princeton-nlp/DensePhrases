from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import load_data, store_data
import string
import argparse


def strip_pred(input_file, gold_file):

    print('original evaluation result:', input_file)
    result = kilt_evaluate(gold=gold_file, guess=input_file)
    print(result)

    preds = load_data(input_file)
    for pred in preds:
        pred['output'][0]['answer'] = pred['output'][0]['answer'].strip(string.punctuation)

    out_file = input_file.replace('.jsonl', '_strip.jsonl')
    print('strip evaluation result:', out_file)
    store_data(out_file, preds)
    new_result = kilt_evaluate(gold=gold_file, guess=out_file)
    print(new_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('gold_file', type=str)
    args = parser.parse_args()
    strip_pred(args.input_file, args.gold_file)
