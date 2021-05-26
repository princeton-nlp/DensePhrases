import json
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_ratio", type=float, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--orig_json", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if (args.sampling_ratio is None) and (args.subset_size is None):
        assert False, "Need to specify either sampling ratio or size of subset!"

    if (args.sampling_ratio is not None):
        assert 0.0 < args.sampling_ratio and args.sampling_ratio < 1.0, \
            "Sampling ratio needs to be between 0 and 1"

    data = json.load(open(args.orig_json))['data']
        
    subset_size = args.subset_size if args.subset_size is not None else \
                  int(round(args.sampling_ratio * len(data)))

    random.seed(args.seed)

    subset = random.sample(data, subset_size)
    with open(args.output_path, 'w') as f:
        json.dump({"data": subset}, f)
