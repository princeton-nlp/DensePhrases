import json
import logging
import numpy as np
from tqdm import tqdm
from densephrases.models import DensePhrases, MIPS
from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
        drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize
from densephrases.utils.embed_utils import get_question_results
from transformers import AutoTokenizer, AutoConfig, MODEL_MAPPING

logger = logging.getLogger(__name__)

def load_data(data_path,
              truecase_path = None,
              draft = False,
              do_lower_case = True,
              shuffle = False):
    q_ids = []
    questions = []
    answers = []
    data = json.load(open(data_path))['data']
    for item in data:
        q_id = item['id']
        question = item['question']
        answer = item['answers']
        if len(answer) == 0:
            continue
        q_ids.append(q_id)
        questions.append(question)
        answers.append(answer)
    questions = [query[:-1] if query.endswith('?') else query for query in questions]

    if truecase_path:
        try:
            logger.info('Loading truecaser for queries')
            truecase = TrueCaser(truecase)
            questions = [truecase.get_true_case(query) if query == query.lower() else query for query in questions]
        except Exception as e:
            print(e)

    if do_lower_case:
        logger.info(f'Lowercasing queries')
        questions = [query.lower() for query in questions]

    if draft:
        q_ids = np.array(q_ids)[:draft_num_examples].tolist()
        questions = np.array(questions)[:draft_num_examples].tolist()
        answers = np.array(answers)[:draft_num_examples].tolist()

    if shuffle:
        qa_pairs = list(zip(q_ids, questions, answers))
        random.shuffle(qa_pairs)
        q_ids, questions, answers = zip(*qa_pairs)
        logger.info(f'Shuffling QA pairs')

    logger.info(f'Loading {len(questions)} questions from {data_path}')
    logger.info(f'Sample Q ({q_ids[0]}): {questions[0]}, A: {answers[0]}')
    return q_ids, questions, answers

def get_phrase_vecs(mips, questions, answers, outs,
                    top_k = 100,
                    regex = False):
    assert mips is not None

    # Get phrase and vectors
    phrase_idxs = [[(out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer'],
        out_['start_vec'], out_['end_vec']) for out_ in out]
        for out in outs
    ]
    for b_idx, phrase_idx in enumerate(phrase_idxs):
        while len(phrase_idxs[b_idx]) < top_k * 2: # two separate top-k from start/end
            phrase_idxs[b_idx].append((-1, 0, 0, '', np.zeros((768)), np.zeros((768))))
        phrase_idxs[b_idx] = phrase_idxs[b_idx][:top_k*2]
    flat_phrase_idxs = [phrase for phrase_idx in phrase_idxs for phrase in phrase_idx]
    doc_idxs = [int(phrase_idx_[0]) for phrase_idx_ in flat_phrase_idxs]
    start_idxs = [int(phrase_idx_[1]) for phrase_idx_ in flat_phrase_idxs]
    end_idxs = [int(phrase_idx_[2]) for phrase_idx_ in flat_phrase_idxs]
    phrases = [phrase_idx_[3] for phrase_idx_ in flat_phrase_idxs]
    start_vecs = [phrase_idx_[4] for phrase_idx_ in flat_phrase_idxs]
    end_vecs = [phrase_idx_[5] for phrase_idx_ in flat_phrase_idxs]

    start_vecs = np.stack(
        # [mips.dequant(mips.offset, mips.scale, start_vec) # Use this for IVFSQ4
        [start_vec
         for start_vec, start_idx in zip(start_vecs, start_idxs)]
    )

    end_vecs = np.stack(
        # [mips.dequant(mips.offset, mips.scale, end_vec) # Use this for IVFSQ4
        [end_vec
         for end_vec, end_idx in zip(end_vecs, end_idxs)]
    )

    zero_mask = np.array([[1] if doc_idx >= 0 else [0] for doc_idx in doc_idxs])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Find targets based on exact string match
    match_fn = drqa_regex_match_score if regex else drqa_exact_match_score # Punctuation included
    targets = [[drqa_metric_max_over_ground_truths(match_fn, phrase[3], answer) for phrase in phrase_idx]
        for phrase_idx, answer in zip(phrase_idxs, answers)]
    targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Reshape
    batch_size = len(answers)
    start_vecs = np.reshape(start_vecs, (batch_size, top_k*2, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, top_k*2, -1))
    return start_vecs, end_vecs, targets

def get_query2vec(query_encoder, tokenizer, batch_size=64, cuda = True, max_query_length=64, debug=False):
    device = 'cuda' if cuda else 'cpu'
    def query2vec(queries):
        question_dataloader, question_examples, query_features = get_question_dataloader(
            queries, tokenizer, max_query_length, batch_size=batch_size
        )
        question_results = get_question_results(
            question_examples, query_features, question_dataloader, device, query_encoder, batch_size=batch_size
        )
        if debug:
            logger.info(f"{len(query_features)} queries: {' '.join(query_features[0].tokens_)}")
        outs = []
        for qr_idx, question_result in enumerate(question_results):
            out = (
                question_result.start_vec.tolist(), question_result.end_vec.tolist(), query_features[qr_idx].tokens_
            )
            outs.append(out)
        return outs
    return query2vec

def get_top_phrases(mips, questions, answers, query_encoder, tokenizer, batch_size, path,
                    top_k = 100, max_answer_length = 10):
    # Search
    step = batch_size
    phrase_idxs = []
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, batch_size=batch_size
    )
    for q_idx in range(0, len(questions), step):
        outs = query2vec(questions[q_idx:q_idx+step])
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx+step], nprobe=256,
            top_k=top_k, return_idxs=True,
            max_answer_length=max_answer_length,
        )
        yield questions[q_idx:q_idx+step], answers[q_idx:q_idx+step], outs

def embed_all_query(questions, query_encoder, tokenizer, batch_size=48):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(questions), batch_size)):
        outs = query2vec(questions[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec

if __name__ == '__main__':
    dump_dir = '/home/ubuntu/dph/outputs/dph-nqsqd-pb2_dev_wiki/dump/phrase'
    index_path = '/home/ubuntu/dph/outputs/dph-nqsqd-pb2_dev_wiki/dump/start/16384_flat_SQ4/index.faiss'
    idx2id_path = '/home/ubuntu/dph/outputs/dph-nqsqd-pb2_dev_wiki/dump/start/16384_flat_SQ4/idx2id.hdf5'
    mips = MIPS(phrase_dump_dir = dump_dir, index_path = index_path, idx2id_path = idx2id_path, cuda=True)

    # from load_qa_pairs
    truecase_path = '/home/ubuntu/dph/data/truecase/english_with_questions.dist' # not entirely sure what this does...
    data_path = '/home/ubuntu/dph/data/open-qa/nq-open/dev_preprocessed.json'
    q_ids, questions, answers = load_data(data_path)

    pretrained_name = 'SpanBERT/spanbert-base-cased'
    cache_dir = '/home/ubuntu/dph/cache'
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        do_lower_case=False,
        cache_dir=cache_dir
    )

    config = AutoConfig.from_pretrained(pretrained_name, cache_dir = cache_dir)
    query_encoder = DensePhrases(config = config, tokenizer = tokenizer, transformer_cls = MODEL_MAPPING[config.__class__]).to('cuda')

    query_vec = embed_all_query(questions, query_encoder, tokenizer)
    
    batch_size = 10
    
    for q_idx in range(0, len(questions), batch_size):
        result = mips.search(
            query_vec[q_idx:q_idx+batch_size],
            q_texts=questions[q_idx:q_idx+batch_size], nprobe=256,
            top_k=100, max_answer_length=10,
        )
        prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result]
        break
    
    # for training
    # dataloader, examples, features = get_question_dataloader(questions, tokenizer, batch_size = 10)
    # svs, evs, tgts = get_phrase_vecs(mips, questions, answers, outs)
