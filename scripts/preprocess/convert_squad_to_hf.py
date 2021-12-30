import json
import argparse
import os


def convert_squad_to_hf(input_file):
    data = json.load(open(input_file))['data']
    outputs = []

    for article in data:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id = qa['id']
                question = qa['question']
                answers = {
                    'text': [answer['text'] for answer in qa['answers']],
                    'answer_start': [answer['answer_start'] for answer in qa['answers']]
                }
                is_impossible = qa['is_impossible']
                # Add more if any

                # Sanity check
                for answer in qa['answers']:
                    assert context[answer['answer_start']:answer['answer_start']+len(answer['text'])] == answer['text']

                outputs.append({
                    'id': str(id),
                    'title': title,
                    'context': context,
                    'question': question,
                    'answers': answers,
                    'is_impossible': is_impossible
                })
    
    output_file = os.path.join(os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0] + '_hf.json')
    print(f'Saving {len(outputs)} (context, question, answer) triples.')
    print('Writing to %s\n'% output_file)
    with open(output_file, 'w') as f:
        json.dump({'data': outputs}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, default=None)
    args = parser.parse_args()
    convert_squad_to_hf(args.input_file)
