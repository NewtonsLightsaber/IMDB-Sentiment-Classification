# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from models import BernoulliNaiveBayes as BNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
from train import models_path, reports_path, num_bnb_features, get_train_test_data

def main():
    data_names = (
        'X_train.json',
        'X_test.json',
        'y_train.json',
    )
    X_train, X_test, y_train = get_train_test_data(interim_path, data_names)

    model_names = (
        'BernoulliNaiveBayes.pkl',
        'LogisticRegression.pkl',
        'SupportVectorMachine.pkl'
    )
    models = get_models(models_path, model_names)

    predict_train(models, X_train, y_pred)
    predict_test(models, [name.split('.')[0] for name in model_names], X_test)

def predict_train(models, X_train, y_train):
    for model in models:
        if isinstance(model, GridSearchCV):
            report(model.cv_results_)
        else:
            X = X_train if not isintance(model, BNB) else X_train[:num_bnb_features]
            y_train_pred = model.predict(X)
            print(metrics.classification_report(
                y_train, y_train_pred))

def predict_test(models, model_names, X_test):
    for model, name in zip(models, model_names):
        y_test_pred = model.predict(X_test)

        # Export to CSV file
        pd.DataFrame(y_test_pred,
            columns=['Category']).to_csv(reports_path / (name + '.csv'))

def get_models(input_path, filenames):
    return [get_model(input_path / filename) for filename in filenames]

def get_model(path):
    return pickle.load(open(path, 'rb'))

def report(results, n_top=3):
    """
    Helper method to find the highest ranking models
    From: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
