import argparse
import json
import os
import random
import logging
import pickle
import torch
import faiss
import h5py
import numpy as np
from tqdm import tqdm

from densephrases.utils.embed_utils import int8_to_float
from densephrases import Options

logger = logging.getLogger(__name__)


def get_args():
    options = Options()
    options.add_index_options()
    args = options.parse()

    coarse = 'hnsw' if args.hnsw else 'flat'
    args.index_name = f'{args.num_clusters}_{coarse}_{args.fine_quant}{"_first" if args.first_passage else ""}'
    if args.index_filter != -1e8: # other than default
        args.index_name = args.index_name + f'_ft{int(args.index_filter)}'
    args.index_dir = os.path.join(args.dump_dir, 'start', args.index_name)

    args.quantizer_path = os.path.join(args.index_dir, args.quantizer_path)
    args.trained_index_path = os.path.join(args.index_dir, args.trained_index_path)
    args.inv_path = os.path.join(args.index_dir, args.inv_path)

    args.subindex_dir = os.path.join(args.index_dir, args.subindex_name)
    if args.dump_paths is None:
        args.index_path = os.path.join(args.index_dir, args.index_path)
        args.idx2id_path = os.path.join(args.index_dir, args.idx2id_path)
    else:
        args.dump_paths = [os.path.join(args.dump_dir, args.phrase_dir, path) for path in args.dump_paths.split(',')]
        args.index_path = os.path.join(args.subindex_dir, '%d.faiss' % args.offset)
        args.idx2id_path = os.path.join(args.subindex_dir, '%d.hdf5' % args.offset)

    logger.info(f"Creating {args.index_name}...")
    return args


def concat_vectors(vectors):
    total_size = sum(vec.shape[0] for vec in vectors)
    if len(vectors[0].shape) > 1:
        out_vector = np.zeros((total_size, *vectors[0].shape[1:]), dtype=vectors[0].dtype)
    else:
        out_vector = np.zeros((total_size), dtype=vectors[0].dtype)
    vec_idx = 0
    for vec in vectors:
        out_vector[vec_idx:vec_idx+vec.shape[0]] = vec
        vec_idx += vec.shape[0]
    return out_vector


def sample_data(dump_paths, doc_sample_ratio=0.2, vec_sample_ratio=0.2, seed=29, norm_th=999):
    start_vecs = []
    end_vecs = []
    random.seed(seed)
    np.random.seed(seed)
    print('sampling from:')
    for dump_path in dump_paths:
        print(dump_path)
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    for i, f in enumerate(tqdm(dumps)):
        doc_ids = list(f.keys())
        sampled_doc_ids = random.sample(doc_ids, int(doc_sample_ratio * len(doc_ids)))
        for doc_id in tqdm(sampled_doc_ids, desc='sampling from %d' % i):
            doc_group = f[doc_id]
            groups = [doc_group]
            for group in groups:
                start_set = group['start'][:]
                if len(start_set.shape) < 2:
                    continue
                num_start, d = start_set.shape
                if num_start == 0: continue
                sampled_start_idxs = np.random.choice(num_start, int(vec_sample_ratio * num_start))
                start_vec = int8_to_float(start_set, group.attrs['offset'], group.attrs['scale'])[sampled_start_idxs]
                start_vec = start_vec[np.linalg.norm(start_vec, axis=1) <= norm_th]
                start_vecs.append(start_vec)

    start_out = concat_vectors(start_vecs)
    for dump in dumps:
        dump.close()

    avg_vec = np.mean(start_out, axis=0, keepdims=True)
    std_vec = np.std(start_out, axis=0, keepdims=True)

    return start_out, avg_vec, std_vec


def train_index(start_data, quantizer_path, trained_index_path, num_clusters,
        fine_quant='SQ4', cuda=False, hnsw=False):
    ds = start_data.shape[1]
    quantizer = faiss.IndexFlatIP(ds)

    # Used only for reimplementation
    if fine_quant == 'SQ4':
        start_index = faiss.IndexIVFScalarQuantizer(
            quantizer, ds, num_clusters, faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_INNER_PRODUCT
        )

    # Default index type
    elif 'OPQ' in fine_quant:
        code_size = int(fine_quant[fine_quant.index('OPQ')+3:])
        if hnsw:
            start_index = faiss.IndexHNSWPQ(ds, "HNSW32,PQ96", faiss.METRIC_INNER_PRODUCT)
        else:
            opq_matrix = faiss.OPQMatrix(ds, code_size)
            opq_matrix.niter = 10
            sub_index = faiss.IndexIVFPQ(quantizer, ds, num_clusters, code_size, 8, faiss.METRIC_INNER_PRODUCT)
            start_index = faiss.IndexPreTransform(opq_matrix, sub_index)
    elif 'none' in fine_quant:
        start_index = faiss.IndexFlatIP(ds)
    else:
        raise ValueError(fine_quant)

    start_index.verbose = False
    if cuda:
        # Convert to GPU index
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
        gpu_index.verbose = False

        # Train on GPU and back to CPU
        gpu_index.train(start_data)
        start_index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        start_index.train(start_data)

    # Make sure to set direct map again
    if 'none' not in fine_quant:
        index_ivf = faiss.extract_index_ivf(start_index)
        index_ivf.make_direct_map()
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
    faiss.write_index(start_index, trained_index_path)


def add_with_offset(start_index, start_data, start_valids, start_total, offset, fine_quant):
    if 'none' in fine_quant:
        start_index.add(start_data)
    else:
        start_ids = (np.arange(start_data.shape[0]) + offset + start_total).astype(np.int64)
        start_index.add_with_ids(start_data, start_ids)

    if len(start_valids) != sum(start_valids):
        print('start invalid')


def add_to_index(dump_paths, trained_index_path, target_index_path, idx2id_path,
                 num_docs_per_add=1000, cuda=False, fine_quant='SQ4', offset=0, norm_th=999,
                 ignore_ids=None, avg_vec=None, std_vec=None, first_passage=False, index_filter=-1e8):

    sidx2doc_id = []
    sidx2word_id = []
    dumps = [h5py.File(dump_path, 'r') for dump_path in dump_paths]
    # filter dumps
    if index_filter != -1e8:
        f_dumps = [h5py.File(dump_path.replace('/phrase/', '/filter/'), 'r') for dump_path in dump_paths]

    print('reading %s' % trained_index_path)
    start_index = faiss.read_index(trained_index_path)
    if 'none' not in fine_quant:
        index_ivf = faiss.extract_index_ivf(start_index)
        index_ivf.make_direct_map()
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)

    if cuda:
        if 'PQ' in fine_quant:
            index_ivf = faiss.extract_index_ivf(start_index)
            quantizer = index_ivf.quantizer
            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer)
            index_ivf.quantizer = quantizer_gpu
        else:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            start_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)

    print('adding following dumps:')
    for dump_path in dump_paths:
        print(dump_path)
    start_total = 0
    start_total_prev = 0
    cnt = 0
    for di, phrase_dump in enumerate(tqdm(dumps, desc='dumps')):
        starts = []
        start_valids = []
        dump_length = len(phrase_dump)
        for i, (doc_idx, doc_group) in enumerate(tqdm(phrase_dump.items(), desc='adding %d' % di)):
            if ignore_ids is not None and doc_idx in ignore_ids:
                continue
            num_start = doc_group['start'].shape[0]
            if num_start == 0: continue
            cnt += 1

            # First passage only
            if first_passage:
                f2o_start = doc_group['f2o_start'][:]
                cut = sum(f2o_start < doc_group['len_per_para'][0])
                start = int8_to_float(
                    doc_group['start'][:cut], doc_group.attrs['offset'], doc_group.attrs['scale']
                )
                num_start = start.shape[0]
            # Apply index filter
            elif index_filter != -1e8:
                o2f_start = {orig: ft for ft, orig in enumerate(doc_group['f2o_start'][:])}
                filter_start = f_dumps[di][doc_idx]['filter_start'][:] 
                filter_end = f_dumps[di][doc_idx]['filter_end'][:] 
                start_idxs, = np.where(filter_start > index_filter)
                end_idxs, = np.where(filter_end > index_filter)
                save_idx = set(np.concatenate([start_idxs, end_idxs]))
                save_idx = sorted([o2f_start[si] for si in save_idx if si in o2f_start])
                start = int8_to_float(
                    doc_group['start'][save_idx], doc_group.attrs['offset'], doc_group.attrs['scale']
                )
                num_start = start.shape[0]
            else:
                start = int8_to_float(
                    doc_group['start'][:], doc_group.attrs['offset'], doc_group.attrs['scale']
                )
            start_valid = np.linalg.norm(start, axis=1) <= norm_th

            starts.append(start)
            start_valids.append(start_valid)
            sidx2doc_id.extend([int(doc_idx)] * num_start)
            if index_filter == -1e8:
                sidx2word_id.extend(range(num_start))
            else:
                sidx2word_id.extend(save_idx)
            start_total += num_start

            if len(starts) > 0 and ((i % num_docs_per_add == 0) or (i == dump_length - 1)):
                print('adding at %d' % (i+1))
                add_with_offset(
                    start_index, concat_vectors(starts), concat_vectors(start_valids), start_total_prev, offset, fine_quant,
                )
                start_total_prev = start_total
                starts = []
                start_valids = []
        if len(starts) > 0:
            print('final adding at %d' % (i+1))
            add_with_offset(
                start_index, concat_vectors(starts), concat_vectors(start_valids), start_total_prev, offset, fine_quant,
            )
            start_total_prev = start_total
    print('number of docs', cnt)

    for dump in dumps:
        dump.close()

    if cuda:
        print('moving back to cpu')
        if 'PQ' in fine_quant:
            index_ivf.quantizer = quantizer
            del quantizer_gpu
        else:
            start_index = faiss.index_gpu_to_cpu(start_index)

    print('start_index ntotal: %d' % start_index.ntotal)
    print(start_total)
    sidx2doc_id = np.array(sidx2doc_id, dtype=np.int32)
    sidx2word_id = np.array(sidx2word_id, dtype=np.int32)

    print('writing index and metadata')
    with h5py.File(idx2id_path, 'w') as f:
        g = f.create_group(str(offset))
        g.create_dataset('doc', data=sidx2doc_id)
        g.create_dataset('word', data=sidx2word_id)
        g.attrs['offset'] = offset

    faiss.write_index(start_index, target_index_path)
    print('done')


def merge_indexes(subindex_dir, trained_index_path, target_index_path, target_idx2id_path, target_inv_path):
    # target_inv_path = merged_index.ivfdata
    names = os.listdir(subindex_dir)
    idx2id_paths = [os.path.join(subindex_dir, name) for name in names if name.endswith('.hdf5')]
    index_paths = [os.path.join(subindex_dir, name) for name in names if name.endswith('.faiss')]
    print(len(idx2id_paths))
    print(len(index_paths))

    print('copying idx2id')
    with h5py.File(target_idx2id_path, 'w') as out:
        for idx2id_path in tqdm(idx2id_paths, desc='copying idx2id'):
            with h5py.File(idx2id_path, 'r') as in_:
                for key, g in in_.items():
                    offset = str(g.attrs['offset'])
                    assert key == offset
                    group = out.create_group(offset)
                    group.create_dataset('doc', data=in_[key]['doc'])
                    group.create_dataset('word', data=in_[key]['word'])

    print('loading invlists')
    ivfs = []
    for index_path in tqdm(index_paths, desc='loading invlists'):
        # the IO_FLAG_MMAP is to avoid actually loading the data thus
        # the total size of the inverted lists can exceed the
        # available RAM
        index = faiss.read_index(index_path,
                                 faiss.IO_FLAG_MMAP)
        ivfs.append(index.invlists)

        # avoid that the invlists get deallocated with the index
        index.own_invlists = False

    # construct the output index
    index = faiss.read_index(trained_index_path)

    # prepare the output inverted lists. They will be written
    # to merged_index.ivfdata
    invlists = faiss.OnDiskInvertedLists(
        index.nlist, index.code_size,
        target_inv_path)

    # merge all the inverted lists
    print('merging')
    ivf_vector = faiss.InvertedListsPtrVector()
    for ivf in tqdm(ivfs):
        ivf_vector.push_back(ivf)

    print("merge %d inverted lists " % ivf_vector.size())
    ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
    print(ntotal)

    # now replace the inverted lists in the output index
    index.ntotal = ntotal
    index.replace_invlists(invlists)

    print('writing index')
    faiss.write_index(index, target_index_path)


def run_index(args):
    dump_names = os.listdir(os.path.join(args.dump_dir, args.phrase_dir))
    dump_paths = sorted(
        [os.path.join(args.dump_dir, args.phrase_dir, name) for name in dump_names if name.endswith('.hdf5')]
    )

    data = None
    if args.stage in ['all', 'coarse']:
        if args.replace or not os.path.exists(args.quantizer_path):
            if not os.path.exists(args.index_dir):
                os.makedirs(args.index_dir)
            start_data, avg_vec, std_vec = sample_data(
                dump_paths, doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio,
                norm_th=args.norm_th
            )
            with open(os.path.join(args.index_dir, 'avg_vec.pkl'), 'wb') as fp:
                pickle.dump(avg_vec, fp)
            with open(os.path.join(args.index_dir, 'std_vec.pkl'), 'wb') as fp:
                pickle.dump(std_vec, fp)

    if args.stage in ['all', 'fine']:
        if args.replace or not os.path.exists(args.trained_index_path):
            if start_data is None:
                start_data, avg_vec, std_vec = sample_data(
                    dump_paths,
                    doc_sample_ratio=args.doc_sample_ratio, vec_sample_ratio=args.vec_sample_ratio,
                    norm_th=args.norm_th,
                    hnsw=args.hnsw
                )
            train_index(
                start_data, args.quantizer_path, args.trained_index_path, args.num_clusters,
                fine_quant=args.fine_quant, cuda=args.cuda, hnsw=args.hnsw
            )

    if args.stage in ['all', 'add']:
        if args.replace or not os.path.exists(args.index_path):
            avg_vec = None
            std_vec = None
            # with open(os.path.join(args.index_dir, 'avg_vec.pkl'), 'rb') as fp:
            #     avg_vec = pickle.load(fp)
            # with open(os.path.join(args.index_dir, 'std_vec.pkl'), 'rb') as fp:
            #     std_vec = pickle.load(fp)

            if args.dump_paths is not None:
                dump_paths = args.dump_paths
                if not os.path.exists(args.subindex_dir):
                    os.makedirs(args.subindex_dir)
            add_to_index(
                dump_paths, args.trained_index_path, args.index_path, args.idx2id_path,
                cuda=args.cuda, num_docs_per_add=args.num_docs_per_add, offset=args.offset, norm_th=args.norm_th,
                fine_quant=args.fine_quant, avg_vec=avg_vec, std_vec=std_vec,
                first_passage=args.first_passage, index_filter=args.index_filter,
            )

    if args.stage == 'merge':
        if args.replace or not os.path.exists(args.index_path):
            merge_indexes(args.subindex_dir, args.trained_index_path, args.index_path, args.idx2id_path, args.inv_path)

    if args.stage == 'move':
        index = faiss.read_index(args.trained_index_path)
        invlists = faiss.OnDiskInvertedLists(
            index.nlist, index.code_size,
            args.inv_path)
        index.replace_invlists(invlists)
        faiss.write_index(index, args.index_path)


def main():
    args = get_args()
    run_index(args)


if __name__ == '__main__':
    main()
