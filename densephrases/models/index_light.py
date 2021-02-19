import logging
import numpy as np
from time import time

from densephrases.models.index import MIPS


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MIPSLight(MIPS):
    def __init__(self, *args, **kwargs):
        logger.info("Loading a light version of MIPS")
        super(MIPSLight, self).__init__(*args, **kwargs)

    def search_dense(self, query, q_texts, top_k, nprobe):
        batch_size, d = query.shape
        self.index.nprobe = nprobe

        # Stack start/end and benefit from multi-threading
        start_time = time()
        query = query.astype(np.float32)
        query_start, query_end = np.split(query, 2, axis=1)
        query_concat = np.concatenate((query_start, query_end), axis=0)

        # Search with faiss
        b_scores, I = self.index.search(query_concat, top_k)
        b_start_scores, start_I = b_scores[:batch_size,:], I[:batch_size,:]
        b_end_scores, end_I = b_scores[batch_size:,:], I[batch_size:,:]
        logger.debug(f'1) {time()-start_time:.3f}s: MIPS')

        # Get idxs from resulting I
        start_time = time()
        b_start_doc_idxs, b_start_idxs = self.get_idxs(start_I, max_idx=1e8)
        b_end_doc_idxs, b_end_idxs = self.get_idxs(end_I, max_idx=1e8)

        # Number of unique docs
        num_docs = sum(
            [len(set(s_doc.flatten().tolist() + e_doc.flatten().tolist())) for s_doc, e_doc in zip(b_start_doc_idxs, b_end_doc_idxs)]
        ) / batch_size
        self.num_docs_list.append(num_docs)
        logger.debug(f'2) {time()-start_time:.3f}s: get index')

        # Merge start/end documents
        start_time = time()
        doc_idxs = [[] for _ in range(batch_size)]
        start_idxs = [[] for _ in range(batch_size)]
        end_idxs = [[] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]
        for b_idx, (s_doc_idxs, s_idxs, s_scores) in enumerate(zip(b_start_doc_idxs, b_start_idxs, b_start_scores)):
            matching_idxs = np.where((np.expand_dims(s_doc_idxs, 1) == np.expand_dims(b_end_doc_idxs[b_idx], 0)) > 0)
            assert all(s_doc_idxs[matching_idxs[0]] == b_end_doc_idxs[b_idx][matching_idxs[1]])
            doc_idxs[b_idx].extend(s_doc_idxs[matching_idxs[0]].tolist())
            start_idxs[b_idx].extend(s_idxs[matching_idxs[0]].tolist())
            end_idxs[b_idx].extend(b_end_idxs[b_idx][matching_idxs[1]].tolist())
            scores[b_idx].extend(s_scores[matching_idxs[0]] + b_end_scores[b_idx][matching_idxs[1]])

        max_idx = max([len(doc_idx) for doc_idx in doc_idxs])
        for doc_idx, start_idx, end_idx, score in zip(doc_idxs, start_idxs, end_idxs, scores):
            while len(doc_idx) != max_idx:
                doc_idx.append(-1)
                start_idx.append(-1)
                end_idx.append(-1)
                score.append(-1e8)
            assert len(doc_idx) == len(start_idx) == len(end_idx) == len(score)

        logger.debug(f'3) {time()-start_time:.3f}s: match docs')
        return doc_idxs, start_idxs, start_I, None, end_idxs, end_I, scores, None

    def search_phrase(self, query, doc_idxs, start_idxs, orig_start_idxs, tmp_doc_idxs, end_idxs, orig_end_idxs, scores,
            tmp_scores, top_k=10, max_answer_length=10, return_idxs=False):

        # Reshape for phrases
        num_queries = query.shape[0]
        q_idxs = np.reshape(np.tile(np.expand_dims(np.arange(num_queries), 1), [1, top_k]), [-1])
        doc_idxs = np.reshape(doc_idxs, [-1])
        start_idxs = np.reshape(start_idxs, [-1])
        end_idxs = np.reshape(end_idxs, [-1])
        scores = np.reshape(scores, [-1])
        assert len(doc_idxs) == len(start_idxs) == len(end_idxs) == len(scores)

        # Get query_end and groups
        start_time = time()
        self.open_dumps()
        groups = {
            doc_idx: self.get_doc_group(doc_idx)
            for doc_idx in set(doc_idxs.tolist()) if doc_idx >= 0
        }
        groups = {
            doc_idx: {
                'word2char_start': groups[doc_idx]['word2char_start'][:],
                'word2char_end': groups[doc_idx]['word2char_end'][:],
                'f2o_start': groups[doc_idx]['f2o_start'][:],
                'f2o_end': groups[doc_idx]['f2o_start'][:],
                'context': groups[doc_idx].attrs['context'],
                'title': groups[doc_idx].attrs['title'],
                'offset': -2,
                'scale': 20,
            }
            for doc_idx in set(doc_idxs.tolist()) if doc_idx >= 0
        }
        self.close_dumps()
        logger.debug(f'1) {time()-start_time:.3f}s: disk access')

        # Filter max or not in start2end
        start_time = time()
        for bb, (doc_idx, start_idx, end_idx) in enumerate(zip(doc_idxs, start_idxs, end_idxs)):
            if (doc_idx >= 0) and (groups[doc_idx]['f2o_end'][end_idx] - groups[doc_idx]['f2o_start'][start_idx] > max_answer_length):
                end_idxs[bb] = int(-1e8)
                scores[bb] = int(-1e8)
            if (doc_idx >= 0) and (groups[doc_idx]['f2o_end'][end_idx] - groups[doc_idx]['f2o_start'][start_idx] < 0):
                end_idxs[bb] = int(-1e8)
                scores[bb] = int(-1e8)

        max_scores = scores
        doc_idxs = np.stack(doc_idxs)
        start_idxs = np.stack(start_idxs)
        end_idxs = np.stack(end_idxs)
        max_scores = np.stack(max_scores)

        out = [{
            'context': groups[doc_idx]['context'], 'title': groups[doc_idx]['title'], 'doc_idx': doc_idx,
            'start_pos': groups[doc_idx]['word2char_start'][groups[doc_idx]['f2o_start'][start_idx]].item(),
            'end_pos': (groups[doc_idx]['word2char_end'][groups[doc_idx]['f2o_end'][end_idx]].item()
                if (len(groups[doc_idx]['word2char_end']) > 0) and (end_idx >= 0)
                else groups[doc_idx]['word2char_start'][groups[doc_idx]['f2o_start'][start_idx]].item() + 1),
            'start_idx': start_idx, 'end_idx': end_idx, 'score': score, 'doc_score': doc_score
            } if doc_idx >= 0 else {
                'score': -1e8, 'context': 'dummy', 'start_pos': 0, 'end_pos': 0}
            for doc_idx, start_idx, end_idx, score in zip(
                doc_idxs.tolist(), start_idxs.tolist(), end_idxs.tolist(), max_scores.tolist())
        ]

        for each in out:
            each['answer'] = each['context'][each['start_pos']:each['end_pos']]
        out = [self.adjust(each) for each in out]

        # Sort output
        new_out = [[] for _ in range(num_queries)]
        for idx, each_out in zip(q_idxs, out):
            new_out[idx].append(each_out)
        for i in range(len(new_out)):
            new_out[i] = sorted(new_out[i], key=lambda each_out: -each_out['score'])
            new_out[i] = list(filter(lambda x: x['score'] > -1e5, new_out[i])) # In case of no output but masks
        logger.debug(f'2) {time()-start_time:.3f}s: get metadata')
        return new_out
