# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import binarize
from make_dataset import NEGATIVE, POSITIVE

class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier with 2 classes.
    """
    def __init__(self, binarize=.0, k=1):
        """
        Set:
            k value for Laplacian Smoothing
            binarize threshold: values = 1 if values > threshold else 0
        """
        self.binarize = binarize
        self.k = k

    def fit(self, X, y):
        """
        Input:
            X: n*m csr_matrix
            y: list of length n
        """
        k = self.k
        n, m = X.shape[0], X.shape[1]
        X = binarize(X, threshold=self.binarize)
        num_y_0 = y.count(0)
        num_y_1 = n - num_y_0

        """
        Define
            theta_1 = (# of examples where y=1) / (total # of examples)
            theta_j_1 = (# examples with xj=1 and y=1) / (# examples with y=1)
            theta_j_0 = (# examples with xj=1 and y=0) / (# examples with y=0)
        Then
            self.theta_x_1[j] = theta_j_1
            self.theta_x_0[j] = theta_j_0
        """
        self.theta_1 = num_y_1 / n
        self.theta_x_0 = csr_matrix(np.full([1,m], k), dtype=np.float64)
        self.theta_x_1 = csr_matrix(np.full([1,m], k), dtype=np.float64)

        for i in range(n):
            if y[i] == NEGATIVE:
                self.theta_x_0 += X[i]

            else: # y[i] == POSITIVE
                self.theta_x_1 += X[i]

        self.theta_x_0 /= float(num_y_0 + k + 1)
        self.theta_x_1 /= float(num_y_1 + k + 1)

        return self

    def predict(self, X):
        """
        Closed form solution for probability.
        Between probabilites of review being persitive versus negative,
        choose the outcome with higher probability.
        """
        n, m = X.shape[0], X.shape[1]
        X = binarize(X, threshold=self.binarize)
        y_pred = np.full(n, POSITIVE)
        theta_1 = self.theta_1
        theta_x_0 = self.theta_x_0.toarray()
        theta_x_1 = self.theta_x_1.toarray()

        prob_pos = theta_1 * np.prod(theta_x_1)
        prob_neg = (1 - theta_1) * np.prod(theta_x_0)

        x_prev = csr_matrix(np.full(m,1))

        for i in range(n):
            x = X[i]
            temp_prob_pos = prob_pos
            temp_prob_neg = prob_neg

            for j in range(m):
                if X[i,j] == 0:
                    temp_prob_pos = temp_prob_pos / theta_x_1[0,j] * (1 - theta_x_1[0,j])
                    temp_prob_neg = temp_prob_neg / theta_x_0[0,j] * (1 - theta_x_0[0,j])

            if temp_prob_pos < temp_prob_neg:
                y_pred[i] = NEGATIVE

        return y_pred
