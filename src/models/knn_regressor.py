from ctypes import Union

import numpy as np

from data.dataset import Dataset
from metrics.rmse import rmse
from statisticsbla.euclidean_distance import euclidean_distance


class KNNRegressor:
    def __init__(self, k: int, distance: euclidean_distance = euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset):
        """
        Stores the dataset.
        :param dataset: Dataset object
        :return: The dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, x: np.ndarray):
        """
        Calculates the mean of the class with the highest frequency.
        :param x: Array of samples.
        :return: Indexes of the classes with the highest frequency
        """

        # Calculates the distance between the samples and the dataset
        distances = self.distance(x, self.dataset.x)

        # Sort the distances and get indexes
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array
        knn_labels = self.dataset.y[knn]

        # Computes the mean of the matching classes
        match_class_mean = np.mean(knn_labels)

        # Sorts the classes with the highest mean
        # high_freq_class = np.argsort(match_class_mean)[:self.k]

        return match_class_mean

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class with the highest frequency
        :return: Class with the highest frequency.
        """
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.x)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)