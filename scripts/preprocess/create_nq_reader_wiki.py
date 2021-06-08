import json
import argparse
import os
import glob
import copy

from tqdm import tqdm


def nq_to_wiki(input_file, output_dir, wiki_dump):
    with open(input_file, 'r') as f:
        nq_data = json.load(f)['data']

    para_cnt = 0
    match_cnt = 0
    title_not_found_cnt = 0
    answer_not_found_cnt = 0
    tokenize_error = 0
    WINDOW = 10
    new_data = []
    for article in tqdm(nq_data):
        title = article['title'] if type(article['title']) != list else article['title'][0]

        assert len(article['paragraphs']) == 1
        for paragraph in article['paragraphs']:
            para_cnt += 1
            new_paragraph = None
            answer_found = False
            assert len(paragraph['qas']) == 1, 'NQ only has single para for each Q'

            # We skip these cases and use existing paras
            qa = paragraph['qas'][0]
            if 'redundant' in str(qa['id']):
                break

            if qa['is_impossible'] or (title not in wiki_dump):
                pass
            else:
                # Or we find matching answers
                answers = qa['answers'] if type(qa['answers']) == list else [qa['answers']]
                for answer in answers:
                    start_window = WINDOW if WINDOW < answer['answer_start'] else answer['answer_start']

                    answer_text = paragraph['context'][
                        answer['answer_start']:answer['answer_start']+len(answer['text'])
                    ].replace('\'\'', '"').replace('``', '"').replace(' ', '').lower()

                    answer_text_with_context = [
                        paragraph['context'][ # Front/Back 10 chars
                            answer['answer_start']-start_window:answer['answer_start']+len(answer['text'])+WINDOW
                        ].replace('\'\'', '"').replace('``', '"').replace(' ', '').lower(),
                        paragraph['context'][ # Front 10 chars
                            answer['answer_start']-start_window:answer['answer_start']+len(answer['text'])
                        ].replace('\'\'', '"').replace('``', '"').replace(' ', '').lower(),
                        paragraph['context'][ # Back 10 chars
                            answer['answer_start']:answer['answer_start']+len(answer['text'])+WINDOW
                        ].replace('\'\'', '"').replace('``', '"').replace(' ', '').lower(),
                    ]

                    new_start = None
                    wiki_paragraph = None
                    for wiki_par in wiki_dump[title]:
                        wiki_par_char = ''.join([char.lower()[0] for char in wiki_par['context'].replace(' ', '')])
                        nosp_to_sp = {}
                        for sp_idx, char in enumerate(wiki_par['context']):
                            if char != ' ':
                                nosp_to_sp[len(nosp_to_sp)] = sp_idx
                        assert len(nosp_to_sp) == len(wiki_par_char)

                        # Context match
                        if any([at_with_context in wiki_par_char for at_with_context in answer_text_with_context]):
                            at_with_context = [at for at in answer_text_with_context if at in wiki_par_char][0]
                            tmp_start = wiki_par_char.index(at_with_context)
                            if len([at for at in answer_text_with_context if at in wiki_par_char]) < 3:
                                if at_with_context == answer_text: # There are some false negatives but we skip
                                    # print(paragraph['context'])
                                    # print(wiki_par['context'])
                                    # print(answer_text)
                                    # import pdb; pdb.set_trace()
                                    break
                            # try:
                            new_start = nosp_to_sp[wiki_par_char[tmp_start:].index(answer_text)+tmp_start]
                            new_end = nosp_to_sp[wiki_par_char[tmp_start:].index(answer_text)+tmp_start+len(answer_text)-1]
                            wiki_paragraph = copy.deepcopy(wiki_par['context'])
                            # except ValueError as e:
                                # print("Could not found start position after de-tokenize")
                                # tokenize_error += 1
                                # import pdb; pdb.set_trace()
                                # continue
                            answer_found = True
                            break
                        # elif answer_text in wiki_par_char:
                        #     answer_found = True

                    # If answer is found, append
                    if new_start is not None:
                        if answer_text != wiki_par['context'][new_start:new_end+1].lower().replace(' ', ''):
                            print('mismatch between original vs. new answer: {} vs. {}'.format(
                                answer_text, wiki_par['context'][new_start:new_end+1].lower().replace(' ', '')
                            ))

                        if new_paragraph is None:
                            new_paragraph = copy.deepcopy(paragraph)
                            new_paragraph['context'] = wiki_paragraph
                            new_paragraph['qas'][0]['answers'] = [{
                                'text': wiki_paragraph[new_start:new_end+1],
                                'answer_start': new_start,
                                'wiki_matched': True,
                            }]
                        else:
                            if new_paragraph['context'] != wiki_paragraph: # If other answers are in different para, we skip
                                continue
                            new_paragraph['qas'][0]['answers'].append({
                                'text': wiki_paragraph[new_start:new_end+1],
                                'answer_start': new_start,
                                'wiki_matched': True,
                            })

            # Just use existing paragraph when no answer is found
            if not answer_found:
                answer_not_found_cnt += 1
                new_paragraph = copy.deepcopy(paragraph)
                for qas in new_paragraph['qas']:
                    for ans in qas['answers']:
                        ans['wiki_matched'] = False
            else:
                match_cnt += 1

            assert new_paragraph is not None
            new_data.append({
                'title': title,
                'paragraphs': [new_paragraph],
            })

    print(f'matched title: {para_cnt}')
    print(f'not found title: {title_not_found_cnt}')
    print(f'matched answer: {match_cnt}')
    print(f'answer not found: {answer_not_found_cnt}')
    print(f'tokenize error: {tokenize_error}')
    print(f'total saved data: {len(new_data)}')
    
    output_path = os.path.join(
        os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0] + '_wiki3.json'
    )
    print(f'Saving into {output_path}')
    with open(output_path, 'w') as f:
        json.dump({'data': new_data}, f)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', type=str, default=None)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('wiki_dir', type=str, default=None)
    args = parser.parse_args()

    # Prepare wiki first
    wiki_files = sorted(glob.glob(args.wiki_dir + "*"))
    print(f'Matching with {len(wiki_files)} number of wikisquad files')
    wiki_dump = {}
    for wiki_file in tqdm(wiki_files):
        with open(wiki_file, 'r') as f:
            wiki_squad = json.load(f)
            for wiki_article in wiki_squad['data']:
                wiki_dump[wiki_article['title']] = wiki_article['paragraphs']
        # break

    for input_file in args.input_files.split(','):
        print(f'Processing {input_file}')
        nq_to_wiki(input_file, args.output_dir, wiki_dump)
