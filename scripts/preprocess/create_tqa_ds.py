
import json
import pdb
import re
import random
from tqdm import tqdm
import string
import argparse

try:
    from eval_utils import (
        drqa_exact_match_score,
        drqa_regex_match_score,
        drqa_metric_max_over_ground_truths
    )
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from eval_utils import (
        drqa_exact_match_score,
        drqa_regex_match_score,
        drqa_metric_max_over_ground_truths
    )
    
# fix random seed
random.seed(0)

def find_substring_and_return_random_idx(substring, string):
    substring_idxs = [m.start() for m in re.finditer(re.escape(substring), string)]
    substring_idx = random.choice(substring_idxs)
    return substring_idx

def main(args):
    print("loading input data")
    with open(args.input_path, encoding='utf-8') as f:
        data = json.load(f)

    output_data = []

    for sample_id in tqdm(data):
        sample = data[sample_id]
        
        question = sample['question']
        answers = sample['answer']
        predictions = sample['prediction']
        titles = sample['title']
        evidences = sample['evidence']
    
        match_fn = drqa_regex_match_score if args.regex else drqa_exact_match_score
        
        answer_text = ""
        answer_start = -1
        ds_context = ""
        ds_title = ""
        # is_from_context = False
        
        # check if prediction is matched in a golden answer in the answer list
        for pred_idx, pred in enumerate(predictions):
            if pred != "" and drqa_metric_max_over_ground_truths(match_fn, pred, answers):
                answer_text = pred
                ds_context = evidences[pred_idx]
                ds_title = titles[pred_idx][0]
                answer_start = find_substring_and_return_random_idx(answer_text, ds_context)
                break
        
        # NOTE! hide these lines because is_from_context contains too many noises
        # # in case prediction is not matched to any golden answer,
        # # check if golden answer is contained in the context
        # if answer_start < 0:
        #     found = False
        #     for evid_idx, evid in enumerate(evidences):
        #         for ans in answers:
        #             if ans != "" and ans in evid:
        #                 found = True
        #                 answer_text = ans
        #                 answer_start = find_substring_and_return_random_idx(ans, evid)
        #                 ds_context = evidences[evid_idx]
        #                 ds_title = titles[evid_idx][0]
        #                 is_from_context = True
        #         if found:
        #             break
        
        # no answer is found in
        is_impossible = False
        if answer_start < 0 or answer_text == "":
            ds_title = titles[0][0]
            ds_context = evidences[0]
            is_impossible = True
        else:
            assert answer_text == ds_context[answer_start:answer_start+len(answer_text)]
        
        output_data.append({
            'title': ds_title,
            'paragraphs':[{
                'context': ds_context,
                'qas':[{
                    'question': question,
                    'is_impossible' : is_impossible,
                    'answers': [{
                        'text': answer_text,
                        'answer_start': answer_start
                    }] if is_impossible == False else [],
                    # 'is_from_context':is_from_context
                }],
                'id': sample_id
            }]
        })
        
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'data': output_data
        },f)
    

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, default='/home/pred/sbcd_sqdqgnqqg_inb64_s384_sqdnq_pinb2_0_20181220_concat_train_preprocessed_78785.pred')
    parser.add_argument('output_path', type=str, default='tqa_ds_train.json')
    parser.add_argument('--regex', action='store_true')
    args = parser.parse_args()

    main(args)
