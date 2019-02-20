# -*- coding: utf-8 -*-
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from make_dataset import project_dir, processed_path
from models import BernoulliNaiveBayes as BNB

models_path = project_dir / 'models'
num_bnb_features = 1000

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
    X_train, X_test, y_train = get_train_test_data(processed_path, filenames)

    #bnb, lr, svm = train_models(X_train, y_train)
    svm = train_svm(X_train, y_train)
    logger.info(('initial training completed, '
                 'now performing cross validation'))

    model_params_pairs = (
        (svm, {
            "vect__ngram_range": [(1,1),(1,2),(1,3),(1,4)],
            "clf__kernel": [
                'linear',
                #'sigmoid',
            ],
            "clf__gamma": [
                #'auto',
                'scale',
            ]
        }),
    )
    svm = grid_search(model_params_pairs)
    logger.info(('cross validation completed, '
                 'now saving models'))

    model_name_pairs = (
        #(bnb, 'BernoulliNaiveBayes.pkl'),
        #(lr, 'LogisticRegression.pkl'),
        (svm, 'SupportVectorMachine.pkl'),
    )
    save_models(model_name_pairs, models_path)
    logger.info('models saved to {0}'.format(models_path))

def grid_search(model_params_pairs, cv=2, verbose=10, return_train_score=True):
    grid_searches = []

    for model in models:
        grid_search = GridSearchCV(
            model,
            param_grid=params,
            cv=cv,
            verbose=verbose,
            return_train_score=return_train_score,
        )
        grid_search.fit(X_train, y_train)
        grid_searches.append(grid_search)

    return grid_searches if len(grid_searches) > 1 else grid_searches[0]

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

    bnb = BNB().fit(X_train_counts[:num_bnb_features], y_train)
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
        ('clf', SVC(cache_size=7000)),
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
