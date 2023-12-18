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
        n_samples, n_features = dataset.X.shape   #number of samples -> rows -> 1000, number of features -> columns -> 5 (exemplo notebook)
        n_classes = len(np.unique(dataset.y))    #number of unique classes -> labels -> 2 (0 or 1)

        # Initialize counts and probabilities
        class_counts = np.zeros(n_classes)     #array de zeros com o tamanho do numero de classes (apenas 1 linha) -> 2
        feature_counts = np.zeros((n_classes, n_features))     #array de zeros com o tamanho do numero de classes e numero de features (2 linhas e 5 colunas)
        class_prior = np.zeros(n_classes)   #array de zeros com o tamanho do numero de classes (apenas 1 linha)

        # Compute counts
        for i in range(n_samples):     # percorre todas as linhas do dataset (todas as 1000 samples)
            class_counts[dataset.y[i]] += 1   # conta o numero de vezes que aparece cada classe (0 ou 1)
            feature_counts[dataset.y[i]] += dataset.X[i]   # conta o numero de vezes que aparece cada feature (0 ou 1) para cada classe (0 ou 1)

        # Apply Laplace smoothing
        class_counts += self.smoothing * n_classes  # soma o smoothing a cada classe (0 ou 1)
        feature_counts += self.smoothing    # soma o smoothing a cada feature para cada classe 

        # Compute class_prior
        class_prior = class_counts / (n_samples + self.smoothing * n_classes)  

        # Compute feature_probs
        feature_probs = feature_counts / class_counts[:, np.newaxis] #probabilidade para cada feature para cada classe

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
            class_probs = np.prod(dataset.X[i] * self.feature_probs + (1 - dataset.X[i]) * (1 - self.feature_probs), axis=1) * self.class_prior #probabilidade de cada classe para cada feature
            predicted_class = np.argmax(class_probs) #classe com maior probabilidade
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