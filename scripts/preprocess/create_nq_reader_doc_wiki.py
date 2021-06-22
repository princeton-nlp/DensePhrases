import json
import glob
import argparse
from tqdm import tqdm
import os

def main(args):
    wiki_dir = args.wiki_dir
    nq_dir = args.nq_reader_docs_dir
    output_dir = args.output_dir
    
    wiki_file_list = glob.glob(os.path.join(wiki_dir,"*"))
    wiki_titles = []
    num_wiki = 0
    wiki_title2paragraphs = {}
    for filename in tqdm(wiki_file_list, total=len(wiki_file_list)):
        with open(filename,'r') as f:
            data = json.load(f)['data']
            
        for doc in data:
            title = doc['title']
            wiki_titles.append(title)
            paragraph = doc['paragraphs']
            wiki_title2paragraphs[title] = paragraph
            num_wiki += 1

    assert len(wiki_title2paragraphs) == num_wiki

    nq_file_list = glob.glob(os.path.join(nq_dir,"*"))
    nq_titles = []
    unmatched_titles = []
    num_matched = 0
    num_unmatched = 0
    for filename in tqdm(nq_file_list, total=len(nq_file_list)):
        with open(filename,'r') as f:
            data = json.load(f)['data']
            
        for doc in data:
            title = doc['title']
            nq_titles.append(title)
            if title in wiki_title2paragraphs:
                doc['paragraphs'] = wiki_title2paragraphs[title]
                num_matched += 1
            else:
                unmatched_titles.append(title)
                num_unmatched +=1
            
            new_paragraphs = []
            for paragraph in doc['paragraphs']:
                if ('is_paragraph' in paragraph) and (not paragraph['is_paragraph']):
                    continue
                
                new_paragraphs.append({
                    'context': paragraph['context']
                })
            doc['paragraphs'] = new_paragraphs
                    
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        output_path = os.path.join(output_dir,os.path.basename(filename))
        output = {
            'data': data
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
            
        # with open('unmatched_title.txt', 'w') as f:
        #     for title in unmatched_titles:
        #         if 'list of' in title:
        #             continue
        #         f.writelines(title)
        #         f.writelines("\n")

    print("num_matched={} num_unmatched={}".format(num_matched, num_unmatched))
    print("len(nq_titles)={} len(wiki_titles)={}".format(len(nq_titles), len(wiki_titles)))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--wiki_dir", type=str, required=True)
    parser.add_argument("--nq_reader_docs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)

