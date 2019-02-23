# -*- coding: utf-8 -*-
import logging
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from data import processed_path
from models import BernoulliNaiveBayes as BNB
from train import get_train_test_data

def main():
    logger = logging.getLogger(__name__)
    logger.info('Testing BNB')

    filenames = (
        'X_train.json',
        'X_test.json',
        'y_train.json',
    )
    X_train, X_test, y_train = get_train_test_data(processed_path, filenames)

    count_vect = CountVectorizer().fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
    X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)

    test_bnb(X_train_normalized, y_train)

def test_bnb(X_train, y_train):
    m = 60
    x_max = X_train[:,:m].max()
    print(x_max)

    for binarize in np.arange(start=0, stop=x_max, step=0.01):
        bnb = BNB(binarize=binarize).fit(X_train[:,:m], y_train)
        y_train_pred = bnb.predict(X_train[:,:m])

        print('Binarize threshold: {0}'.format(binarize))
        print(metrics.classification_report(
            y_train, y_train_pred))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
