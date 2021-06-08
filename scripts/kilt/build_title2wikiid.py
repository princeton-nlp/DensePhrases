#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
import unicodedata
import html

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', html.unescape(text))

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    # documents = []
    results = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            
            title = normalize(doc['title'])
            if '&amp' in title:
                import pdb; pdb.set_trace()
                
            if 'u0' in title:
                import pdb; pdb.set_trace()
            results[title] = doc['id']
    return results


def store_contents(data_path, save_path):
    results = {}
    files = [f for f in iter_files(data_path)]
    for file in tqdm(files):
        contents = get_contents(file)
        results.update(contents)
    
    print(f"len(results)={len(results)}")
    with open(save_path, 'w') as f:
        json.dump(results, f)

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='/path/to/data')
    parser.add_argument('--save_path', type=str, help='/path/to/saved/db.db')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path
    )