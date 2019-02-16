# -*- coding: utf-8 -*-
import json
import numpy as np
from pathlib import Path
#from sklearn.datasets import fetch_20newsgroups
#newsgroups = fetch_20newsgroups(subset='all')
from make_dataset import interim_path, processed_path
from models import BernoulliNaiveBayes as BNB

def main():
    train_set = get_dataset(interim_path / 'train.json')
    #X_train = get_X_bnb(train_set)
    y_train = get_y(train_set)

    #bnb = BNB()
    #bnb.train()

def get_dataset(path):
    return json.load(open(path))

def get_y(dataset):
    return np.array([[ptn['sentiment']] for ptn in dataset])

def get_X_bnb(dataset):
    pass

if __name__ == '__main__':
    main()
