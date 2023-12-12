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
        n_samples, n_features = dataset.X.shape
        n_classes = len(np.unique(dataset.y))

        # Initialize counts and probabilities
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        class_prior = np.zeros(n_classes)

        # Compute counts
        for i in range(n_samples):
            class_counts[dataset.y[i]] += 1
            feature_counts[dataset.y[i]] += dataset.X[i]

        # Apply Laplace smoothing
        class_counts += self.smoothing * n_classes
        feature_counts += self.smoothing

        # Compute class_prior
        class_prior = class_counts / (n_samples + self.smoothing * n_classes)

        # Compute feature_probs
        feature_probs = feature_counts / class_counts[:, np.newaxis]

        self.class_prior = class_prior
        self.feature_probs = feature_probs

        return self

    def predict(self, dataset):
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
        predictions = []

        for i in range(len(dataset.X)):
            class_probs = np.prod(dataset.X[i] * self.feature_probs + (1 - dataset.X[i]) * (1 - self.feature_probs), axis=1) * self.class_prior
            predicted_class = np.argmax(class_probs)
            predictions.append(predicted_class)

        return np.array(predictions)

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