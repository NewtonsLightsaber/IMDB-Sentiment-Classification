# -*- coding: utf-8 -*-
import logging
import json
from pathlib import Path
from zipfile import ZipFile

project_dir = Path(__file__).resolve().parents[1]
raw_path = project_dir / 'data' / 'raw'
interim_path = project_dir / 'data' / 'interim'

NEGATIVE = 0
POSITIVE = 1

def main():
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final datasets from raw data')
    preprocess(raw_path, interim_path) # Group all data into json datasets

def preprocess(input_path, output_path):
    zipfiles = [ 'train.zip', 'test.zip' ]

    for zipfile in zipfiles:
        set_name = zipfile.split('.')[0]
        X, y = [], []

        with ZipFile(input_path / zipfile) as zip:
            txtfiles = (
                name
                for name in zip.namelist()
                if '.txt' in name and 'MACOSX' not in name # Ignore 'MACOSX' directory
            )
            for txtfile in txtfiles:
                text = zip.read(txtfile).decode('utf-8')
                X.append(text)

                # Add sentiment label if in training dataset
                if set_name == 'train':
                    y.append(POSITIVE if 'pos' in txtfile else NEGATIVE)

        with open(output_path / ('X_' + set_name + '.json'), 'w') as fout:
            json.dump(X, fout)

        if set_name == 'train':
            with open(output_path / ('y_' + set_name + '.json'), 'w') as fout:
                json.dump(y, fout)

if __name__ == '__main__':
    main()
