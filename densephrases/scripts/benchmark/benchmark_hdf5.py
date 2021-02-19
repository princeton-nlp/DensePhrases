import h5py

from tqdm import tqdm


paths = [
    'dumps/sbcd_sqdqgnqqg_inb64_s384_sqdnq_pinb2_0_20181220_concat/dump/phrase/0-200.hdf5',
    'dumps/sbcd_sqdqgnqqg_inb64_s384_sqdnq_pinb2_0_20181220_concat/dump/phrase/200-400.hdf5'
]
phrase_dumps = [h5py.File(path, 'r') for path in paths]


# Just testing how fast it is to read hdf5 files from disk
for phrase_dump in phrase_dumps:
    for doc_id, doc_val in tqdm(phrase_dump.items()):
        kk = doc_val['start'][-10:]
