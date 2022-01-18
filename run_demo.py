import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging

from tqdm import tqdm
from time import time
from flask import Flask, request, jsonify, render_template, redirect
from flask_cors import CORS
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from requests_futures.sessions import FuturesSession

from eval_phrase_retrieval import evaluate_results, evaluate_results_kilt
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.data_utils import TrueCaser
from densephrases.utils.open_utils import load_phrase_index, load_cross_encoder, load_qa_pairs, get_query2vec
from densephrases import Options


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DensePhrasesDemo(object):
    def __init__(self, args):
        self.args = args
        self.base_ip = args.base_ip
        self.query_port = args.query_port
        self.index_port = args.index_port
        self.truecase = TrueCaser(os.path.join(os.environ['DATA_DIR'], args.truecase_path))

    def serve_query_encoder(self, query_port, args, inmemory=False, batch_size=64, query_encoder=None, tokenizer=None):
        device = 'cuda' if args.cuda else 'cpu'
        if query_encoder is None:
            query_encoder, tokenizer, _ = load_encoder(device, args, query_only=True)
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
            # start_time = time()
            outs = list(query2vec(batch_query))
            # logger.info(f'query2vec {time()-start_time:.3f} for {len(batch_query)} queries: {batch_query[0]}')
            return jsonify(outs)

        logger.info(f'Starting QueryEncoder server at {self.get_address(query_port)}')
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(query_port)
        IOLoop.instance().start()

    def serve_phrase_index(self, index_port, args):
        args.examples_path = os.path.join('densephrases/demo/static', args.examples_path)

        # Load mips
        mips = load_phrase_index(args)
        app = Flask(__name__, static_folder='./densephrases/demo/static/')
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        CORS(app)

        def batch_search(batch_query, max_answer_length=20, top_k=10,
                         nprobe=64, return_vecs=False):
            t0 = time()
            outs, _ = self.embed_query(batch_query)()
            start = np.concatenate([out[0] for out in outs], 0)
            end = np.concatenate([out[1] for out in outs], 0)
            query_vec = np.concatenate([start, end], 1)

            rets = mips.search(
                query_vec, q_texts=batch_query, nprobe=nprobe,
                top_k=top_k, max_answer_length=max_answer_length,
                return_vecs=return_vecs, aggregate=True,
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

    def batch_query(self, batch_query, batch_context=None, max_answer_length=20, top_k=10, nprobe=64):
        post_data = {
            'query': json.dumps(batch_query),
            'context': json.dumps(batch_context) if batch_context is not None else json.dumps(batch_query),
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
        qids, questions, answers, _ = load_qa_pairs(args.test_path, args)

        # Run batch_query and evaluate
        step = args.eval_batch_size
        predictions = []
        evidences = []
        titles = []
        scores = []
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
            predictions += prediction
            evidences += evidence
            titles += title
            scores += score
        logger.info(f'{time()-start_time:.3f} sec for {num_q} questions => {num_q/(time()-start_time):.1f} Q/Sec')
        eval_fn = evaluate_results if not args.is_kilt else evaluate_results_kilt
        eval_fn(
            predictions, qids, questions, answers, args, evidences=evidences, scores=scores, titles=titles,
        )


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_demo_options()
    args = options.parse()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    server = DensePhrasesDemo(args)

    if args.run_mode == 'q_serve':
        logger.info(f'Query address: {server.get_address(server.query_port)}')
        server.serve_query_encoder(args.query_port, args)

    elif args.run_mode == 'p_serve':
        logger.info(f'Index address: {server.get_address(server.index_port)}')
        server.serve_phrase_index(args.index_port, args)

    elif args.run_mode == 'query':
        query = 'Name three famous writers'
        result = server.query(query)
        logger.info(f'Answers to a question: {query}')
        logger.info(f'{[r["answer"] for r in result["ret"]]}')

    elif args.run_mode == 'batch_query':
        queries = [
            'Where',
            'when did medicare begin in the united states',
            'who sings don\'t stand so close to me',
            'Name three famous writers',
            'Who was defeated by computer in chess game?'
        ]
        contexts = [
            'Uncle jesse\'s original full name was James Lee. And he was born in South Korea.',
            'Uncle jesse\'s original name was James Lee. And he was born in South Korea in 1333. US medicare started in 1222.',
            'Uncle jesse\'s original name was James Lee. And he sang this song.',
            'Uncle jesse\'s original name was James Lee. And Jens wrote this novel.',
            'Uncle jesse\'s original name was James Lee. The man was defeated by Alphago.'
        ]
        result = server.batch_query(
            queries,
            contexts, # feed context for cross encoders
            max_answer_length=args.max_answer_length,
            top_k=args.top_k,
            nprobe=args.nprobe,
        )
        for query, re in zip(queries, result['ret'].values()):
            logger.info(f'Answers to a question: {query}')
            logger.info(f'{re}')

    elif args.run_mode == 'eval_request':
        server.eval_request(args)

    else:
        raise NotImplementedError
