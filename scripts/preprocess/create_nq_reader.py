import json
import argparse
import pdb
import glob
from nq_utils import load_examples

def convert_tokens_to_answer(paragraph_tokens, answer_tokens):
    answer_token_indexes = []
    for answer_token in answer_tokens:
        answer_token_index = paragraph_tokens.index(answer_token)
        answer_token_indexes.append(answer_token_index)
    
    if len(answer_token_indexes) != (answer_token_indexes[-1] - answer_token_indexes[0] + 1):
        print("answer_token_indexes=",answer_token_indexes)
        pdb.set_trace()
    
    
    context = ""
    answer_text = ""
    answer_start = -1
    for i, paragraph_token in enumerate(paragraph_tokens):
        # skip html token
        if not paragraph_token['html_token']:
            token = paragraph_token['token']
                       
            # prepare appending token with white space     
            if context != "": context +=" "
            
            # update answer_start
            if i == answer_token_indexes[0]:
                answer_start = len(context)
            
            # append token
            context += token
            
            # update answer_end
            if i == answer_token_indexes[-1]:
                answer_end = len(context)

    answer_text = context[answer_start:answer_end]
    
    # sanity check
    assert context != ""
    assert answer_text != ""
    assert answer_start != -1
    
    return context, answer_text, answer_start
    
def main(args):
    # load nq_open and get ids
    with open(args.nq_open_path, 'r') as f:
        nq_open_data = json.load(f)['data']
    nq_open_ids = [qas['id'] for qas in nq_open_data]

    # load nq_orig
    nq_orig_paths = sorted(glob.glob(args.nq_orig_path_pattern))
    nq_reader_data = []
    for i, nq_orig_path in enumerate(nq_orig_paths):
        with open(nq_orig_path, mode='rb') as fileobj:
            examples = load_examples(fileobj, 'train', 'short_answers')

        # filter examples contained in nq_open ids
        examples = dict(filter(lambda x: int(x[0]) in nq_open_ids, list(examples.items())))
        
        for example_id, example in examples.items():
            # filter candidates with answers
            candidates = list(filter(lambda x: x.contains_answer, example.candidates))
            if len(candidates) == 0:
                continue
            
            title = example.title
            # TODO! consider multi annotation for nq_orig_dev set
            short_answers = example.short_answers[0] # assume single annotation
            paragraphs=[]
            
            for candidate in candidates:
                # filter <P> examples
                contents = candidate.contents
                is_paragraph = contents.startswith('<P>')
                start_token = candidate.start_token
                end_token = candidate.end_token
                tokens = example.document_tokens[start_token:end_token]
                
                answers = []
                for short_answer in short_answers:
                    answer_start_token = short_answer['start_token']
                    answer_end_token = short_answer['end_token']
                    if answer_end_token-answer_start_token>5:
                        continue
                    answer_tokens = example.document_tokens[answer_start_token:answer_end_token]
                    # convert tokens to context, answer_text, answer_start
                    context, answer_text, answer_start = convert_tokens_to_answer(tokens, answer_tokens)
                    answers.append({
                        'text': answer_text,
                        'answer_start': answer_start
                    })
                    
                qas = [{
                    'question':example.question_text,
                    'is_impossible': False if is_paragraph else True,
                    'answers':answers,
                    'is_distant': False,
                    'id':int(example_id),
                }]
                paragraphs.append({
                    'context':context,
                    'qas':qas
                })
            nq_reader_data.append({
                'title': title,
                'paragraphs':paragraphs
            })
    
    nq_reader = {
        'data' : nq_reader_data
    }
    # save nq_reader
    with open(args.output_path,'w') as f:
        json.dump(nq_reader, f, indent=2)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--nq_open_path",
        default=None,
        type=str,
        required=True,
        help="nq-open path (eg. nq-open/dev.json)"
    )
    parser.add_argument(
        "--nq_orig_path_pattern",
        default=None,
        type=str,
        required=True,
        help="nq-open path (eg. natural-questions/train/nq-train-*.jsonl.gz)"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="nq-reader directory (eg. nq-reader/dev.json)"
    )
    
    args = parser.parse_args()
    
    main(args)

