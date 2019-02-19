# -*- coding: utf-8 -*-
import json
import pickle
import numpy as np
import nltk
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from make_dataset import project_dir, processed_path
from models import BernoulliNaiveBayes as BNB

models_dir = project_dir / 'models'

def main():
    logger = logging.getLogger(__name__)
    logger.info(('training bernoulli naive bayes, '
                 'logistic regression '
                 'and support vector machine models'))

    filenames = (
        'X_train.json',
        'X_test.json',
        'y_train.json',
    )
    X_train, X_test, y_train = get_train_test_data(interim_path, filenames)

    bnb, lr, svm = train_models(X_train, y_train)
    model_name_pairs = (
        (bnb, 'BernoulliNaiveBayes.pkl'),
        (lr, 'LogisticRegression.pkl'),
        (bnb, 'SupportVectorMachine.pkl'),
    )
    save_models(model_name_pairs, models_dir)

def save_models(model_name_pairs, output_path):
    for model, name in model_name_pairs:
        pickle.dump(model, open(output_path / name, 'wb'))

def train_models(X_train, y_train):
    bnb = train_bnb(X_train, y_train)
    lr = train_lr(X_train, y_train)
    svm = train_svm(X_train, y_train)

    return bnb, lr, svm

def train_bnb(X_train, y_train):
    count_vect = CountVectorizer().fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_test_counts = count_vect.transform(X_test)

    bnb = BNB().fit(X_train_counts, y_train)
    return bnb

def train_lr(X_train, y_train):
    pclf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        ('clf', LR()),
    ])
    pclf.fit(X_train, y_train)
    return pclf

def train_svm(X_train, y_train):
    pclf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('norm', Normalizer()),
        #('clf', SVC(kernel='rbf', gamma='auto')),
        #('clf', SVC(kernel='poly')),
        ('clf', SVC(kernel='linear')),
    ])
    pclf.fit(X_train, y_train)
    return pclf

def get_train_test_data(input_path, filenames):
    train_test_data = [
        get_dataset(input_path / filename)
        for filename in filenames
    ]
    return train_test_data

def get_dataset(path):
    return json.load(open(path))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
