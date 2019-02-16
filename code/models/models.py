# -*- coding: utf-8 -*-
import numpy as np
from ...data.make_dataset import NEGATIVE, POSITIVE

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier with 2 classes.
    """
    def train(self, X, y):
        """
        Define
            theta_1 = (# of examples where y=1) / (total # of examples)
            theta_j_1 = (# examples with xj=1 and y=1) / (# examples with y=1)
            theta_j_0 = (# examples with xj=1 and y=0) / (# examples with y=0)
        Then
            self.theta_x_1[j] = theta_j_1
            self.theta_x_0[j] = theta_j_0
        """
        n = X.shape[0]
        m = X.shape[1]
        X, y = X.tolist(), flatten(y.tolist())
        num_y_0 = y.count([0])
        num_y_1 = n - num_y_0

        self.theta_1 = num_y_1 / n
        self.theta_x_0 = [
            (len([None for i in range(n) if X[i][j] == 1 and y[i] == 0]) + 1) \
            / (num_y_0 + 2)
            for j in range(m)
        ]
        self.theta_x_1 = [
            (len([None for i in range(n) if X[i][j] == y[i] == 1]) + 1) \
            / (num_y_1 + 2)
            for j in range(m)
        ]

        return self

    def predict(self, X):
        m = X.shape[1]
        X = X.tolist()
        predictions = []

        for x in X:
            prob_y_1 = np.prod(
                  [self.theta_1]
                + [self.theta_x_1[j]
                    if x[j] == 1 else 1 - self.theta_x_1[j]
                        for j in range(m)]
            )
            prob_y_0 = np.prod(
                  [1 - self.theta_1]
                + [self.theta_x_0[j]
                    if x[j] == 1 else 1 - self.theta_x_0[j]
                        for j in range(m)]
            )
            prediction = POSITIVE if prob_y_1 > prob_y_0 else NEGATIVE
            predictions.append([prediction])

        predictions = np.array(predictions)
        return predictions

def flatten(lst):
    """
    Flatten a list of lists
    """
    return [el for sublist in lst for el in sublist]
