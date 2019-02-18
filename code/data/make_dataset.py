# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
from zipfile import ZipFile

project_dir = Path(__file__).resolve().parents[2]
raw_path = project_dir / 'data' / 'raw'
interim_path = project_dir / 'data' / 'interim'
processed_path = project_dir / 'data' / 'processed'

NEGATIVE = 0
POSITIVE = 1

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final datasets from raw data')

    format(raw_path, interim_path)

def preprocess(input_path, output_path):
    pass

def format(input_path, output_path):

    zipfiles = [ 'train.zip', 'test.zip' ]

    for zipfile in zipfiles:
        dataset = []

        with ZipFile(input_path / zipfile) as zip:
            txtfiles = (
                name
                for name in zip.namelist()
                if '.txt' in name and 'MACOSX' not in name # Ignore 'MACOSX' directory
            )

            for txtfile in txtfiles:
                text = zip.read(txtfile).decode('utf-8')

                datapoint = { 'text': text }
                
                # Add sentiment label if known
                if 'train' in txtfile:
                    datapoint['sentiment'] = POSITIVE if 'pos' in txtfile else NEGATIVE

                dataset.append(datapoint)

        with open(output_path / (zipfile.split('.')[0] + '.json'), 'w') as fout:
            json.dump(dataset, fout)

if __name__ == '__main__':
    main()