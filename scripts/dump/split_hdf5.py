import h5py
import os
from tqdm import tqdm

input_dump_dir = 'dumps/sbcd_sqd_ftinb84_kl_x4_20181220_concat/dump/phrase/'
select = 6
print(f'************** {select} *****************')
input_dump_paths = sorted(
    [os.path.join(input_dump_dir, name) for name in os.listdir(input_dump_dir) if 'hdf5' in name]
)[select:select+1]
print(input_dump_paths)
input_dumps = [h5py.File(path, 'r') for path in input_dump_paths]

dump_names = [os.path.splitext(os.path.basename(path))[0] for path in input_dump_paths]
dump_ranges = [list(map(int, name.split('-'))) for name in dump_names]
new_ranges = []
for range_ in dump_ranges:
    # print(range_)
    middle = sum(range_) // 2 # split by half
    new_range_ = [[range_[0], middle], [middle, range_[1]]]
    # print(new_range_)
    new_ranges.append(new_range_)

output_dumps = [
    [h5py.File(f'dumps/sbcd_sqd_ftinb84_kl_x4_20181220_concat/dump/phrase/{ra[0]}-{ra[1]}.hdf5', 'w')
        for ra in range_]
    for range_ in new_ranges
]

print(input_dumps)
print(output_dumps)
print(new_ranges)

# dev-100M-c 160408
# dev_wiki_noise 250000

for dump_idx, (input_dump, new_range, output_dump) in tqdm(enumerate(zip(input_dumps, new_ranges, output_dumps))):
    print(f'splitting {input_dump} to {output_dump}')
    for idx, (key, val) in tqdm(enumerate(input_dump.items())):
        # if idx < 250000/2:
        if int(key) < new_range[0][1] * 1000:
            output_dump[0].copy(val, key)
        else:
            output_dump[1].copy(val, key)

    input_dump.close()
    output_dump[0].close()
    output_dump[1].close()

print('copy done')
