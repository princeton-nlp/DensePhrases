import json
import argparse
import os
import csv

from tqdm import tqdm
# from drqa.retriever.utils import normalize

def get_gold_answers_kilt(gold):
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
    return ground_truths

def preprocess_openqa(input_file, input_type, out_dir):
    data_to_save = []
    # SQuAD
    if input_type == 'SQuAD':
        with open(input_file, 'r') as f:
            articles = json.load(f)['data']
            for article in articles:
                for paragraph in article['paragraphs']:
                    for qa in paragraph['qas']:
                        if type(qa['answers']) == dict:
                            qa['answers'] = [qa['answers']]
                        data_to_save.append({
                            'id': qa['id'],
                            'question': qa['question'],
                            'answers': [ans['text'] for ans in qa['answers']]
                        })
    # CuratedTrec / WebQuestions / WikiMovies
    elif input_type == 'DrQA':
        tag = os.path.splitext(os.path.basename(input_file))[0]
        for line_idx, line in tqdm(enumerate(open(input_file))):
            data = json.loads(line)
            # answers = [normalize(a) for a in data['answer']] # necessary?
            answers = [a for a in data['answer']]
            data_to_save.append({
                'id': f'{tag}_{line_idx}',
                'question': data['question'],
                'answers': answers
            })
    # NaturalQuestions / TriviaQA
    elif input_type == 'HardEM':
        tag = os.path.splitext(os.path.basename(input_file))[0]
        data = json.load(open(input_file))['data']
        for item_idx, item in tqdm(enumerate(data)):
            data_to_save.append({
                'id': f'{tag}_{item_idx}',
                'question': item['question'],
                'answers': item['answers']
            })
    # DPR style files
    elif input_type == 'DPR':
        tag = os.path.splitext(os.path.basename(input_file))[0]
        data = json.load(open(input_file))
        for item_idx, item in tqdm(enumerate(data)):
            data_to_save.append({
                'id': f'{tag}_{item_idx}',
                'question': item['question'],
                'answers': item['answers']
            })
    # COVID-19
    elif input_type == 'COVID-19':
        assert os.path.isdir(input_file)
        for filename in os.listdir(input_file):
            if 'preprocessed' in filename:
                print(f'Skipping {filename}')
                continue
            file_path = os.path.join(input_file, filename)
            tag = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'r') as f:
                with tqdm(enumerate(f)) as tq:
                    tq.set_description(filename + '\t')
                    for line_idx, line in tq:
                        data_to_save.append({
                            'id': f'{tag}_{line_idx}',
                            'question': line.strip(),
                            'answers': ['']
                        })
    # TREX, ZSRE (KILT)
    elif input_type.lower() in ['trex', 't-rex', 'zsre']:
        with open(input_file) as f:
            for line in tqdm(f):
                data = json.loads(line)
                id = data['id']
                question = data['input']
                answers = get_gold_answers_kilt(data)
                answers = list(answers)
                
                data_to_save.append({
                    'id': id,
                    'question': question,
                    'answers': answers
                })
    # Jsonl (LAMA)
    elif input_type.lower() in ['jsonl']:
        tag = os.path.splitext(os.path.basename(input_file))[0]
        with open(input_file) as f:
            for line_idx, line in tqdm(enumerate(f)):
                data = json.loads(line)
                question = data['question']
                answers = data['answer']
                
                data_to_save.append({
                    'id': f'{tag}_{line_idx}',
                    'question': question,
                    'answers': answers
                })
    # CSV
    elif input_type.lower() in ['csv']:
        import ast
        tag = os.path.splitext(os.path.basename(input_file))[0]
        with open(input_file) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for line_idx, line in tqdm(enumerate(csv_reader)):
                question = line[0]
                answers = ast.literal_eval(line[1])
                
                data_to_save.append({
                    'id': f'{tag}_{line_idx}',
                    'question': question,
                    'answers': answers
                })
    else:
        raise NotImplementedError

    assert os.path.exists(out_dir)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(input_file))[0] + '_preprocessed.json')
    print(f'Saving {len(data_to_save)} questions.')
    print('Writing to %s\n'% out_path)
    with open(out_path, 'w') as f:
        json.dump({'data': data_to_save}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, default=None)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--input_type', type=str, default='SQuAD', help='SQuAD|DrQA|HardEM')
    args = parser.parse_args()
    preprocess_openqa(args.input_file, args.input_type, args.out_dir)
