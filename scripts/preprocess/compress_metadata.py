import pdb
import os
import h5py
from tqdm import tqdm
import sys
import zlib
import numpy as np
import traceback
import blosc
import pickle
import argparse

# get size of the whole metadata
def get_size(d):
    size = 0
    for i in d:
        word2char_start_size = sys.getsizeof(d[i]['word2char_start'])
        word2char_end_size = sys.getsizeof(d[i]['word2char_end'])
        f2o_start_size = sys.getsizeof(d[i]['f2o_start'])
        context_size = sys.getsizeof(d[i]['context'])
        title_size = sys.getsizeof(d[i]['title'])
        size+=word2char_start_size
        size+=word2char_end_size
        size+=f2o_start_size
        size+=context_size
        size+=title_size

    return size

# compress metadata using zlib
# http://python-blosc.blosc.org/tutorial.html
def compress(d):
    for i in d:
        word2char_start = d[i]['word2char_start']
        word2char_end = d[i]['word2char_end']
        f2o_start = d[i]['f2o_start']
        context=d[i]['context']
        title=d[i]['title']

        # save type to use when decompressing
        type1= word2char_start.dtype
        type2= word2char_end.dtype
        type3= f2o_start.dtype
        
        d[i]['word2char_start'] = blosc.compress(word2char_start, typesize=1,cname='zlib')
        d[i]['word2char_end'] = blosc.compress(word2char_end, typesize=1,cname='zlib')
        d[i]['f2o_start'] = blosc.compress(f2o_start, typesize=1,cname='zlib')
        d[i]['context'] = blosc.compress(context.encode('utf-8'),cname='zlib')
        d[i]['dtypes']={
                'word2char_start':type1,
                'word2char_end':type2,
                'f2o_start':type3
        }

        # check if compression is lossless
        try:
            decompressed_word2char_start = np.frombuffer(blosc.decompress(d[i]['word2char_start']), type1)
            decompressed_word2char_end = np.frombuffer(blosc.decompress(d[i]['word2char_end']), type2)
            decompressed_f2o_start = np.frombuffer(blosc.decompress(d[i]['f2o_start']), type3)
            decompressed_context = blosc.decompress(d[i]['context']).decode('utf-8')

            assert ((word2char_start == decompressed_word2char_start).all())
            assert ((word2char_end == decompressed_word2char_end).all())
            assert ((f2o_start ==decompressed_f2o_start).all())
            assert (context == decompressed_context)
        except Exception as e:
            print(e)
            traceback.print_exc()
            pdb.set_trace()
    return d

def load_doc_groups(phrase_dump_dir):
    phrase_dump_paths = sorted(
        [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir) if 'hdf5' in name]
    )
    doc_groups = {}
    types = ['word2char_start', 'word2char_end', 'f2o_start']
    attrs = ['context', 'title']
    phrase_dumps = [h5py.File(path, 'r') for path in phrase_dump_paths]
    phrase_dumps = phrase_dumps[:1]
    for path in tqdm(phrase_dump_paths, desc='loading doc groups'):
        with h5py.File(path, 'r') as f:
            for key in tqdm(f):
                doc_group = {}
                for type_ in types:
                    doc_group[type_] = f[key][type_][:]
                for attr in attrs:
                    doc_group[attr] = f[key].attrs[attr]
                doc_groups[key] = doc_group

    return doc_groups

def main(args):
    # Use it for saving to memory
    doc_groups = load_doc_groups(args.input_dump_dir)

    # Get the size of meta data before compression
    size_before_compression = get_size(doc_groups)

    # compress metadata using zlib
    doc_groups = compress(doc_groups)

    # Get the size of meta data before compression
    size_after_compression = get_size(doc_groups)

    print(f"compressed by {round(size_after_compression/size_before_compression*100,2)}%")

    # save compressed meta as a pickle format
    output_file = os.path.join(args.output_dir, 'meta_compressed.pkl')
    with open(output_file,'wb') as f:
        pickle.dump(doc_groups, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dump_dir', type=str, default='dump/sbcd_sqdqgnqqg_inb64_s384_sqdnq_pinb2_0_20181220_concat/dump/phrase')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()
    
    main(args)
