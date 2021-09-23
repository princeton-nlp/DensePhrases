import os
import random
import logging
import json
import torch
import numpy as np

from densephrases import MIPS
from densephrases.utils.single_utils import backward_compat
from densephrases.utils.squad_utils import get_question_dataloader, TrueCaser
from densephrases.utils.embed_utils import get_question_results

from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
truecase = None


def load_phrase_index(args, ignore_logging=False):
    # Configure paths for index serving
    phrase_dump_dir = os.path.join(args.dump_dir, args.phrase_dir)
    index_dir = os.path.join(args.dump_dir, args.index_name)
    index_path = os.path.join(index_dir, args.index_path)
    idx2id_path = os.path.join(index_dir, args.idx2id_path)

    # Load mips
    if 'aggregate' in args.__dict__.keys():
        logger.info(f'Aggregate: {args.aggregate}')
    mips = MIPS(
        phrase_dump_dir=phrase_dump_dir,
        index_path=index_path,
        idx2id_path=idx2id_path,
        cuda=args.cuda,
        logging_level=logging.WARNING if ignore_logging else (logging.DEBUG if args.verbose_logging else logging.INFO),
    )
    return mips


def load_cross_encoder(device, args):

    # Configure paths for cross-encoder serving
    cross_encoder = torch.load(
        os.path.join(args.load_dir, "pytorch_model.bin"), map_location=torch.device('cpu')
    )
    new_qd = {n[len('bert')+1:]: p for n, p in cross_encoder.items() if 'bert' in n}
    new_linear = {n[len('qa_outputs')+1:]: p for n, p in cross_encoder.items() if 'qa_outputs' in n}
    config, unused_kwargs = AutoConfig.from_pretrained(
        args.pretrained_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        return_unused_kwargs=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.pretrained_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModel.from_pretrained(
        args.pretrained_name_or_path,
        from_tf=bool(".ckpt" in args.pretrained_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.load_state_dict(new_qd)
    qa_outputs = torch.nn.Linear(config.hidden_size, 2)
    qa_outputs.load_state_dict(new_linear)
    ce_model = torch.nn.ModuleList(
        [model, qa_outputs]
    )
    ce_model.to(device)

    logger.info(f'CrossEncoder loaded from {args.load_dir} having {MODEL_MAPPING[config.__class__]}')
    logger.info('Number of model parameters: {:,}'.format(sum(p.numel() for p in ce_model.parameters())))
    return ce_model, tokenizer


def get_query2vec(query_encoder, tokenizer, args, batch_size=64):
    device = 'cuda' if args.cuda else 'cpu'
    def query2vec(queries):
        question_dataloader, question_examples, query_features = get_question_dataloader(
            queries, tokenizer, args.max_query_length, batch_size=batch_size
        )
        question_results = get_question_results(
            question_examples, query_features, question_dataloader, device, query_encoder, batch_size=batch_size
        )
        if args.verbose_logging:
            logger.info(f"{len(query_features)} queries: {' '.join(query_features[0].tokens_)}")
        outs = []
        for qr_idx, question_result in enumerate(question_results):
            out = (
                question_result.start_vec.tolist(), question_result.end_vec.tolist(), query_features[qr_idx].tokens_
            )
            outs.append(out)
        return outs
    return query2vec


def load_qa_pairs(data_path, args, q_idx=None, draft_num_examples=100, shuffle=False):
    q_ids = []
    questions = []
    answers = []
    titles = []
    data = json.load(open(data_path))['data']
    for data_idx, item in enumerate(data):
        if q_idx is not None:
            if data_idx != q_idx:
                continue
        q_id = item['id']
        if 'origin' in item:
            q_id = item['origin'].split('.')[0] + '-' + q_id
        question = item['question']
        if '[START_ENT]' in question:
            question = question[max(question.index('[START_ENT]')-300, 0):question.index('[END_ENT]')+300]
        answer = item['answers']
        title = item.get('titles', [''])
        if len(answer) == 0:
            continue
        q_ids.append(q_id)
        questions.append(question)
        answers.append(answer)
        titles.append(title)
    questions = [query[:-1] if query.endswith('?') else query for query in questions]
    # questions = [query.lower() for query in questions] # force lower query

    if args.do_lower_case:
        logger.info(f'Lowercasing queries')
        questions = [query.lower() for query in questions]

    if shuffle:
        qa_pairs = list(zip(q_ids, questions, answers, titles))
        random.shuffle(qa_pairs)
        q_ids, questions, answers, titles = zip(*qa_pairs)
        logger.info(f'Shuffling QA pairs')

    if args.draft:
        q_ids = np.array(q_ids)[:draft_num_examples].tolist()
        questions = np.array(questions)[:draft_num_examples].tolist()
        answers = np.array(answers)[:draft_num_examples].tolist()
        titles = np.array(titles)[:draft_num_examples].tolist()

    if args.truecase:
        try:
            global truecase
            if truecase is None:
                logger.info('loading truecaser')
                truecase = TrueCaser(os.path.join(os.environ['DATA_DIR'], args.truecase_path))
            logger.info('Truecasing queries')
            questions = [truecase.get_true_case(query) if query == query.lower() else query for query in questions]
        except Exception as e:
            print(e)

    logger.info(f'Loading {len(questions)} questions from {data_path}')
    logger.info(f'Sample Q ({q_ids[0]}): {questions[0]}, A: {answers[0]}, Title: {titles[0]}')
    return q_ids, questions, answers, titles

