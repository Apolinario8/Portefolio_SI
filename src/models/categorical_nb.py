import numpy as np
from typing import List


class CategoricalNB:
    def __init__(self, smoothing: float = 1.0):
        """
        Categorical Naïve Bayes Classifier for binary features.

        Parameters
        ----------
        smoothing: float, optional (default=1.0)
            Laplace smoothing parameter to avoid zero probabilities.
        """
        self.smoothing = smoothing

        self.class_prior = None
        self.feature_probs = None

    def fit(self, dataset):
        """
        Estimate class_prior and feature_probs from the training dataset.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: CategoricalNB
            The fitted model.
        """
        # 1. Define n_samples, n_features, n_classes
        n_samples, n_features = dataset.X.shape
        n_classes = len(np.unique(dataset.y))

        # 2. Initialize class_counts, feature_counts, and class_prior
        self.class_counts = np.zeros(n_classes)
        self.feature_counts = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        # 3. Compute class_counts, feature_counts, and class_prior
        for c in range(n_classes):
            class_mask = (dataset.y == c)
            self.class_counts[c] = np.sum(class_mask)
            self.feature_counts[c, :] = np.sum(dataset.X[class_mask, :], axis=0)

        self.class_prior = self.class_counts / n_samples

        # 4. Apply Laplace smoothing
        self.class_counts += self.smoothing * n_classes
        self.feature_counts += self.smoothing

        # 5. Compute feature_probs
        self.feature_probs = self.feature_counts / self.class_counts[:, np.newaxis]

        return self

    def predict(self, X_test):
        """
        Predict the class labels for a given set of samples.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        predictions: np.ndarray
            The predicted values for the testing dataset.
        """
        X_test = np.array(X_test)

        # Compute the probability for each class for each sample
        class_probs = np.zeros((X_test.shape[0], len(self.class_prior)))

        for c in range(len(self.class_prior)):
            class_probs[:, c] = np.prod(X_test * self.feature_probs[c] + (1 - X_test) * (1 - self.feature_probs[c]), axis=1) * self.class_prior[c]

        # Pick the class with the highest probability as the predicted class
        predictions = np.argmax(class_probs, axis=1)

        return predictions
    def score(self, dataset):
        """
        Calculate the accuracy between estimated classes and actual ones.

        Parameters
        ----------
        dataset: Dataset
            The test dataset.

        Returns
        -------
        accuracy: float
            The accuracy of the model.
        """
        predictions = self.predict(dataset)
        correct_predictions = np.sum(predictions == dataset.y)
        accuracy = correct_predictions / len(dataset.y)
        return accuracy