import argparse
import json
import re
import unicodedata
from collections import defaultdict
from tqdm import tqdm
from scripts.preprocess.simple_tokenizer import SimpleTokenizer


def read_file(infile, handle_file, log=False, skip_first_line=False):
    if log:
        print('Opening "{}"...'.format(infile))
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if log:
        print('  Done.')
    return data


def read_jsonl(infile, log=False):
    handler = lambda f: [json.loads(line) for line in f.readlines()]
    return read_file(infile, handler, log=log)


def read_json(infile, log=False):
    handler = lambda f: json.load(f)
    return read_file(infile, handler, log=log)


def _normalize(text):
    return unicodedata.normalize('NFD', text)

###############################################################################
### HAS_ANSWER FUNCTIONS   ####################################################
###############################################################################
def has_answer_field(ctx, answers):
    return ctx['has_answer']


tokenizer = SimpleTokenizer(**{})
def string_match(ctx, answers):
    text = tokenizer.tokenize(ctx['text']).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False


def normalized_title(ctx, answers):
    for answer in answers:
        a = a.lower().strip()
        title = ctx['title'].lower().strip()
        if a == title[:len(a)]:
            return True
    return False


def regex(ctx, answers):
    text = ctx['text']
    for answer in answers:
        answer = _normalize(answer)
        if regex_match(text, answer):
            return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


###############################################################################
### CALCULATION FUNCTIONS   ###################################################
###############################################################################
def precision_fn(results, k_vals, has_answer):
    n_hits = {k: 0 for k in k_vals}
    mrrs = []
    precs = []
    PREC_K = 20
    MRR_K = 20

    for result in tqdm(results):
        ans = result['answers']
        ctxs = result['ctxs']
        found_k = len(ctxs) + 1
        found = False
        num_hit = 0
        for c_idx,ctx in enumerate(ctxs):
            if has_answer(ctx, ans):
                if not found:
                    found_k = c_idx # record first one
                found = True

                if c_idx < PREC_K: # P@k
                    num_hit += 1
                # break
        for k in k_vals:
            if found_k < k:
                n_hits[k] += 1

        if found_k >= MRR_K:
            mrrs.append(0)
        else:   
            mrrs.append(1/(found_k + 1))
        precs.append(num_hit/PREC_K)
    
    print('*'*50)
    for k in k_vals:
        if len(results) == 0:
            print('No results.')
        else:
            print('Top-{} = {:.2%}'.format(k, n_hits[k] / len(results)))

    print(f'Acc@{k_vals[0]} when Acc@{k_vals[-1]} = {n_hits[k_vals[0]]/n_hits[k_vals[-1]]*100:.2f}%')
    print(f'MRR@{MRR_K} = {sum(mrrs)/len(mrrs)*100:.2f}')
    print(f'P@{PREC_K} = {sum(precs)/len(precs)*100:.2f}')


def precision_fn_file(infile, n_docs, k_vals, has_answer, args):
    results = read_jsonl(infile) if args.jsonl else read_json(infile)

    # stats
    ctx_lens = [sum([len(pp['text'].split()) for pp in re['ctxs']])/len(re['ctxs']) for re in results]
    print(f'ctx token length: {sum(ctx_lens)/len(ctx_lens):.2f}')

    # unique titles
    title_lens = [len(set(pp['title'] for pp in re['ctxs'])) for re in results]
    print(f'unique titles: {sum(title_lens)/len(title_lens):.2f}')

    precision_fn(results, k_vals, has_answer)


# Top-20 and Top-100
def precision_per_bucket(results_file, longtail_file, n_docs, k_vals, longtail_tags, ans_fn):
    results = read_json(results_file)
    annotations = read_json(longtail_file)
    for tag in longtail_tags:
        bucket = [result for idx,result in enumerate(results) if tag == annotations[idx]['annotations']]
        print('==== Bucket={} ====='.format(tag))
        precision_fn(bucket, n_docs, k_vals, ans_fn)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', required=True, type=str, default=None,
                        help="Location of the results file to parse.")
    parser.add_argument('--n_docs', type=int, default=100,
                        help="Maximum number of docs retrieved.")
    parser.add_argument('--k_values', type=str, default='1,5,10,20,40,50,60,80,100',
                        help="Top-K values to print out")
    parser.add_argument('--ans_fn', type=str, default='has_answer',
                        help="How to check whether has the answer. title | has_answer")
    parser.add_argument('--jsonl', action='store_true', help='Set if results is a jsonl file.')

    # Longtail Entity Analysis
    parser.add_argument('--longtail', action='store_true',
                        help='whether or not to include longtail buckets')
    parser.add_argument('--longtail_file', required=False, type=str, default=None,
                        help='Mapping from question to longtail entity tags.')
    parser.add_argument('--longtail_tags', type=str, default='p10,p25,p50,p75,p90',
                        help='Tags for the longtail entities within longtail_file')

    args = parser.parse_args()
    ks = [int(k) for k in args.k_values.split(',')]
    if args.ans_fn == 'has_answer':
        ans_fn = has_answer_field
    elif args.ans_fn == 'title':
        ans_fn = normalized_title
    elif args.ans_fn == 'string':
        ans_fn = string_match
    elif args.ans_fn == 'regex':
        ans_fn = regex
    else:
        raise Exception('Answer function not recognized')
    
    if args.longtail:
        longtail_tags = args.longtail_tags.split(',')
        precision_per_bucket(args.results_file, args.longtail_file, 
            args.n_docs, ks, longtail_tags, ans_fn)
    else:
        precision_fn_file(args.results_file, args.n_docs, ks, ans_fn, args)
