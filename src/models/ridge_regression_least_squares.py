import numpy as np

import sys
sys.path.append(r'C:\Users\gonca\Documents\GitHub\Portefolio_SI\src')

from data.dataset import Dataset
from metrics.mse import mse


class RidgeRegressionLeastSquares:
    def __init__(self, l2_penalty=1.0, scale=True):

        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset):
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape

        # Add intercept term to X
        X = np.c_[np.ones(m), X]

        # Compute the penalty term
        penalty_matrix = self.l2_penalty * np.eye(n + 1)
        penalty_matrix[0, 0] = 0  # Set the first position to 0

        # Compute model parameters using the closed-form solution
        X_transpose = X.T
        inv_term = np.linalg.inv(X_transpose.dot(X) + penalty_matrix)
        self.theta = inv_term.dot(X_transpose).dot(dataset.y)
        self.theta_zero = self.theta[0]
        self.theta = self.theta[1:]

        return self

    def predict(self, dataset):
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape

        # Add intercept term to X
        X = np.c_[np.ones(m), X]

        # Compute predicted Y
        predicted_y = X.dot(np.r_[self.theta_zero, self.theta])

        return predicted_y

    def score(self, dataset):
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)


# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X= X, y= y)

    # fit the model
    print("...................... This code ...........................")
    model = RidgeRegressionLeastSquares(l2_penalty = 2.0, scale=True)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print("Score:", model.score(dataset_))


    print("...................... SKlearn ........................")
    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha = 2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print("Score:", mse(dataset_.y, model.predict(X)))