import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import wandb
import string

from tqdm import tqdm
from time import time
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from requests_futures.sessions import FuturesSession

from densephrases.experiments.run_open import load_query_encoder, load_phrase_index, load_qa_pairs, evaluate_results, \
        get_query2vec, evaluate_results_kilt
from densephrases.utils.squad_utils import get_cq_dataloader, TrueCaser
from densephrases.utils.embed_utils import get_cq_results


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DensePhrasesInterface(object):
    def __init__(self, args):
        self.args = args
        self.base_ip = args.base_ip
        self.query_port = args.query_port
        self.index_port = args.index_port
        self.truecase = TrueCaser(os.path.join(os.environ['DPH_DATA_DIR'], args.truecase_path))

    def serve_query_encoder(self, query_port, args, inmemory=False, batch_size=64, query_encoder=None, tokenizer=None):
        device = 'cuda' if args.cuda else 'cpu'
        if query_encoder is None:
            query_encoder, tokenizer = load_query_encoder(device, args)
        query2vec = get_query2vec(
            query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
        )

        # Serve query encoder
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/query2vec_api', methods=['POST'])
        def query2vec_api():
            batch_query = json.loads(request.form['query'])
            start_time = time()
            outs = query2vec(batch_query)
            # logger.info(f'query2vec {time()-start_time:.3f} for {len(batch_query)} queries: {batch_query[0]}')
            return jsonify(outs)

        logger.info(f'Starting QueryEncoder server at {self.get_address(query_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(query_port)
        IOLoop.instance().start()

    def serve_phrase_index(self, index_port, args):
        args.examples_path = os.path.join('densephrases/demo/static', args.examples_path)

        # Load mips
        mips = load_phrase_index(args, load_light=args.load_light)
        app = Flask(__name__, static_url_path='/static')
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        def batch_search(batch_query, max_answer_length=20, top_k=10,
                         nprobe=64, return_idxs=False):
            t0 = time()
            outs, _ = self.embed_query(batch_query)()
            start = np.concatenate([out[0] for out in outs], 0)
            end = np.concatenate([out[1] for out in outs], 0)
            query_vec = np.concatenate([start, end], 1)

            rets = mips.search(
                query_vec, q_texts=batch_query, nprobe=nprobe,
                top_k=top_k, max_answer_length=max_answer_length,
                return_idxs=return_idxs,
            )
            for ret_idx, ret in enumerate(rets):
                for rr in ret:
                    rr['query_tokens'] = outs[ret_idx][2]
            t1 = time()
            out = {'ret': rets, 'time': int(1000 * (t1 - t0))}
            return out

        @app.route('/')
        def index():
            return app.send_static_file('index.html')

        @app.route('/files/<path:path>')
        def static_files(path):
            return app.send_static_file('files/' + path)

        # This one uses a default hyperparameters (for Demo)
        @app.route('/api', methods=['GET'])
        def api():
            query = request.args['query']
            query = query[:-1] if query.endswith('?') else query
            if args.truecase:
                if query[1:].lower() == query[1:]:
                    query = self.truecase.get_true_case(query)
            out = batch_search(
                [query],
                max_answer_length=args.max_answer_length,
                top_k=args.top_k,
                nprobe=args.nprobe,
            )
            out['ret'] = out['ret'][0]
            return jsonify(out)

        @app.route('/batch_api', methods=['POST'])
        def batch_api():
            batch_query = json.loads(request.form['query'])
            max_answer_length = int(request.form['max_answer_length'])
            top_k = int(request.form['top_k'])
            nprobe = int(request.form['nprobe'])
            out = batch_search(
                batch_query,
                max_answer_length=max_answer_length,
                top_k=top_k,
                nprobe=nprobe,
            )
            return jsonify(out)

        @app.route('/get_examples', methods=['GET'])
        def get_examples():
            with open(args.examples_path, 'r') as fp:
                examples = [line.strip() for line in fp.readlines()]
            return jsonify(examples)

        if self.query_port is None:
            logger.info('You must set self.query_port for querying. You can use self.update_query_port() later on.')
        logger.info(f'Starting Index server at {self.get_address(index_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(index_port)
        IOLoop.instance().start()

    def serve_bert_encoder(self, bert_port, args):
        device = 'cuda' if args.cuda else 'cpu'
        bert_encoder, tokenizer = load_query_encoder(device, args) # will be just a bert as query_encoder
        import binascii
        def float_to_hex(vals):
            strs = []
            # offset = -40.
            # scale = 5.
            minv = min(vals)
            maxv = max(vals)
            for val in vals:
                strs.append('{0:0{1}X}'.format(int(min((val - minv) / (maxv-minv) * 255, 255)), 2))
            return strs

        # Define query to vector function
        def context_query_to_logit(context, query):
            bert_encoder.eval()

            # Phrase encoding style
            dataloader, examples, features = get_cq_dataloader(
                [context], [query], tokenizer, args.max_query_length, batch_size=64
            )
            cq_results = get_cq_results(
                examples, features, dataloader, device, bert_encoder, batch_size=64
            )

            outs = []
            for cq_idx, cq_result in enumerate(cq_results):
                # import pdb; pdb.set_trace()
                all_logits = (
                    np.expand_dims(np.array(cq_result.start_logits), axis=1) +
                    np.expand_dims(np.array(cq_result.end_logits), axis=0)
                ).max(1).tolist()
                out = {
                    'context': ' '.join(features[cq_idx].tokens[0:]),
                    'title': 'dummy',
                    'start_logits': float_to_hex(all_logits[0:len(features[cq_idx].tokens)]),
                    'end_logits': float_to_hex(cq_result.end_logits[0:len(features[cq_idx].tokens)]),
                }
                outs.append(out)

            return outs

        # Serve query encoder
        app = Flask(__name__)
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        @app.route('/')
        def index():
            return app.send_static_file('index_single.html')

        @app.route('/files/<path:path>')
        def static_files(path):
            return app.send_static_file('files/' + path)

        args.examples_path = os.path.join('static', 'examples_context.txt')
        @app.route('/get_examples', methods=['GET'])
        def get_examples():
            with open(args.examples_path, 'r') as fp:
                examples = [line.strip() for line in fp.readlines()]
            return jsonify(examples)

        @app.route('/single_api', methods=['GET'])
        def single_api():
            t0 = time()
            single_context = request.args['context']
            single_query = request.args['query']
            # start_time = time()
            outs = context_query_to_logit(single_context, single_query)
            # logger.info(f'single to logit {time()-start_time}')
            t1 = time()
            out = {'ret': outs, 'time': int(1000 * (t1 - t0))}
            return jsonify(out)

        logger.info(f'Starting BertEncoder server at {self.get_address(bert_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(bert_port)
        IOLoop.instance().start()

    def get_address(self, port):
        assert self.base_ip is not None and len(port) > 0
        return self.base_ip + ':' + port

    def embed_query(self, batch_query):
        emb_session = FuturesSession()
        r = emb_session.post(self.get_address(self.query_port) + '/query2vec_api',
            data={'query': json.dumps(batch_query)})
        def map_():
            result = r.result()
            emb = result.json()
            return emb, result.elapsed.total_seconds() * 1000
        return map_

    def query(self, query):
        params = {'query': query}
        res = requests.get(self.get_address(self.index_port) + '/api', params=params)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {query}')
            logger.info(res.text)
        return outs

    def batch_query(self, batch_query, max_answer_length=20, top_k=10, nprobe=64):
        post_data = {
            'query': json.dumps(batch_query),
            'max_answer_length': max_answer_length,
            'top_k': top_k,
            'nprobe': nprobe,
        }
        res = requests.post(self.get_address(self.index_port) + '/batch_api', data=post_data)
        if res.status_code != 200:
            logger.info('Wrong behavior %d' % res.status_code)
        try:
            outs = json.loads(res.text)
        except Exception as e:
            logger.info(f'no response or error for q {batch_query}')
            logger.info(res.text)
        return outs

    def eval_request(self, args):
        # Load dataset
        qids, questions, answers = load_qa_pairs(args.test_path, args)

        # Run batch_query and evaluate
        step = args.eval_batch_size
        predictions = []
        evidences = []
        titles = []
        scores = []
        all_tokens = []
        start_time = None
        num_q = 0
        for q_idx in tqdm(range(0, len(questions), step)):
            if q_idx >= 5*step: # exclude warmup
                if start_time is None:
                    start_time = time()
                num_q += len(questions[q_idx:q_idx+step])
            result = self.batch_query(
                questions[q_idx:q_idx+step],
                max_answer_length=args.max_answer_length,
                top_k=args.top_k,
                nprobe=args.nprobe,
            )
            prediction = [[ret['answer'] for ret in out] if len(out) > 0 else [''] for out in result['ret']]
            evidence = [[ret['context'] for ret in out] if len(out) > 0 else [''] for out in result['ret']]
            title = [[ret['title'] for ret in out] if len(out) > 0 else [''] for out in result['ret']]
            score = [[ret['score'] for ret in out] if len(out) > 0 else [-1e10] for out in result['ret']]
            q_tokens = [out[0]['query_tokens'] if len(out) > 0 else '' for out in result['ret']]
            predictions += prediction
            evidences += evidence
            titles += title
            scores += score
            all_tokens += q_tokens
        latency = time()-start_time
        logger.info(f'{time()-start_time:.3f} sec for {num_q} questions => {num_q/(time()-start_time):.1f} Q/Sec')
        eval_fn = evaluate_results if not args.is_kilt else evaluate_results_kilt
        eval_fn(
            predictions, qids, questions, answers, args, evidences=evidences, scores=scores, titles=titles, q_tokens=all_tokens,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # QueryEncoder
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument("--pretrained_name_or_path", default='SpanBERT/spanbert-base-cased', type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--do_lower_case", default=False, action='store_true')
    parser.add_argument('--max_query_length', default=64, type=int)
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--query_encoder_path", default='', type=str)
    parser.add_argument("--query_port", default='-1', type=str)

    # PhraseIndex
    parser.add_argument('--dump_dir', default='dump')
    parser.add_argument('--phrase_dir', default='phrase')
    parser.add_argument('--index_dir', default='256_flat_SQ4')
    parser.add_argument('--index_name', default='index.faiss')
    parser.add_argument('--idx2id_name', default='idx2id.hdf5')
    parser.add_argument('--index_port', default='-1', type=str)

    # These can be dynamically changed.
    parser.add_argument('--max_answer_length', default=10, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--nprobe', default=256, type=int)
    parser.add_argument('--truecase', default=False, action='store_true')
    parser.add_argument("--truecase_path", default='truecase/english_with_questions.dist', type=str)

    # KILT
    parser.add_argument('--is_kilt', default=False, action='store_true')
    parser.add_argument('--kilt_gold_path', default='kilt/trex/trex-dev-kilt.jsonl')
    parser.add_argument('--title2wikiid_path', default='wikidump/title2wikiid.json')

    # Serving options
    parser.add_argument('--examples_path', default='examples.txt')

    # Evaluation
    parser.add_argument('--dev_path', default='open-qa/nq-open/dev_preprocessed.json')
    parser.add_argument('--test_path', default='open-qa/nq-open/test_preprocessed.json')
    parser.add_argument('--candidate_path', default=None)
    parser.add_argument('--regex', default=False, action='store_true')
    parser.add_argument('--eval_batch_size', default=10, type=int)

    # Run mode
    parser.add_argument('--base_ip', default='http://127.0.0.1')
    parser.add_argument('--run_mode', default='batch_query')
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--draft', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--load_light', default=False, action='store_true')
    parser.add_argument('--seed', default=1992, type=int)
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    server = DensePhrasesInterface(args)

    if args.run_mode == 'q_serve':
        logger.info(f'Query address: {server.get_address(server.query_port)}')
        server.serve_query_encoder(args.query_port, args)

    elif args.run_mode == 'p_serve':
        logger.info(f'Index address: {server.get_address(server.index_port)}')
        server.serve_phrase_index(args.index_port, args)

    elif args.run_mode == 'single_serve':
        server.serve_bert_encoder(args.query_port, args)

    elif args.run_mode == 'query':
        query = 'Name three famous writers'
        result = server.query(query)
        logger.info(f'Answers to a question: {query}')
        logger.info(f'{[r["answer"] for r in result["ret"]]}')

    elif args.run_mode == 'batch_query':
        queries= [
            'What was uncle jesse\'s original last name on full house',
            'when did medicare begin in the united states',
            'who sings don\'t stand so close to me',
            'Name three famous writers',
            'Who was defeated by computer in chess game?'
        ]
        result = server.batch_query(
            queries,
            max_answer_length=args.max_answer_length,
            top_k=args.top_k,
            nprobe=args.nprobe,
        )
        for query, result in zip(queries, result['ret']):
            logger.info(f'Answers to a question: {query}')
            logger.info(f'{[r["answer"] for r in result]}')

    elif args.run_mode == 'eval_request':
        server.eval_request(args)

    else:
        raise NotImplementedError
