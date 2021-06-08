"""
download wiki dump 20181220 checking md5sum
"""

import os
import json
import urllib.request
import urllib.parse as urlparse
import argparse
import hashlib
import logging
import portalocker
import pdb
from tqdm import tqdm

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()
    return args

def download_file(url, output_dir, size, expected_md5sum=None):
    """
    download file and check md5sum
    """
    logging.info("url={}".format(url))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    bz2file = os.path.join(output_dir, os.path.basename(url))

    lockfile = '{}.lock'.format(bz2file)
    with portalocker.Lock(lockfile, 'w', timeout=60):
        if not os.path.exists(bz2file) or os.path.getsize(bz2file) != size:
            logging.info("Downloading {}".format(bz2file))
            with urllib.request.urlopen(url) as f:
                with open(bz2file, 'wb') as out:
                    for data in tqdm(f, unit='KB'):
                        out.write(data)
            
            # Check md5sum
            if expected_md5sum is not None:
                md5 = hashlib.md5()
                with open(bz2file, 'rb') as infile:
                    for line in infile:
                        md5.update(line)
                if md5.hexdigest() != expected_md5sum:
                    logging.error('Fatal: MD5 sum of downloaded file was incorrect (got {}, expected {}).'.format(md5.hexdigest(), expected_md5))
                    logging.error('Please manually delete "{}" and rerun the command.'.format(tarball))
                    logging.error('If the problem persists, the tarball may have changed, in which case, please contact the SacreBLEU maintainer.')
                    sys.exit(1)
                else:
                    logging.info('Checksum passed: {}'.format(md5.hexdigest()))

def main(args):
    url = 'https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2'
    expected_md5sum = 'ccf875b2af67109fe5b98b5b720ce322'
    size = 15712882238
    
    download_file(
        url=url,
        output_dir=args.output_dir,
        size=size,
        expected_md5sum=expected_md5sum
    )

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    args = parse_args()
    main(args)