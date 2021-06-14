import h5py
import os
from tqdm import tqdm

input_dump_dir = 'dumps/sbcd_sqd_ftinb84_kl_x4_20181220_concat/dump/phrase/'
select = 0
print(f'************** {select} *****************')
input_dump_paths = sorted(
    [os.path.join(input_dump_dir, name) for name in os.listdir(input_dump_dir) if 'hdf5' in name]
)[select:]
print(input_dump_paths)
input_dumps = [h5py.File(path, 'r') for path in input_dump_paths]
dump_names = [os.path.splitext(os.path.basename(path))[0] for path in input_dump_paths]
print(input_dumps)

# Filter dump for a lighter version
'''
output_dumps = [
    h5py.File(f'dumps/densephrases-multi_wiki-20181220/dump/phrase/{k}.hdf5', 'w')
    for k in dump_names
]
print(output_dumps)


for dump_idx, (input_dump, output_dump) in tqdm(enumerate(zip(input_dumps, output_dumps))):
    print(f'filtering {input_dump} to {output_dump}')
    for idx, (key, val) in tqdm(enumerate(input_dump.items())):

        dg = output_dump.create_group(key)
        dg.attrs['context'] = val.attrs['context'][:]
        dg.attrs['title'] = val.attrs['title'][:]
        for k_, v_ in val.items():
            if k_ not in ['start', 'len_per_para', 'start2end']:
                dg.create_dataset(k_, data=v_[:])

    input_dump.close()
    output_dump.close()

print('filter done')
'''

def load_doc_groups(phrase_dump_dir):
    phrase_dump_paths = sorted(
        [os.path.join(phrase_dump_dir, name) for name in os.listdir(phrase_dump_dir) if 'hdf5' in name]
    )
    doc_groups = {}
    types = ['word2char_start', 'word2char_end', 'f2o_start']
    attrs = ['context', 'title']
    phrase_dumps = [h5py.File(path, 'r') for path in phrase_dump_paths]
    for path in tqdm(phrase_dump_paths, desc='loading doc groups'):
        with h5py.File(path, 'r') as f:
            for key in tqdm(f):
                import pdb; pdb.set_trace()
                doc_group = {}
                for type_ in types:
                    doc_group[type_] = f[key][type_][:]
                for attr in attrs:
                    doc_group[attr] = f[key].attrs[attr]
                doc_groups[key] = doc_group
    return doc_groups

# Save below as a pickle file and load it on memory for later use
doc_groups = load_doc_groups(input_dump_dir)
