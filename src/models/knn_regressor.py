from typing import Callable, Union

import numpy as np

from data.dataset import Dataset
from metrics.rmse import rmse
from statisticsbla.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    The k-Nearst Neighbors regressor is a machine learning model that predicts the value of new samples based on a similarity measure (e.g., distance functions).
    This algorithm predicts the value of new samples by looking at the values of the k-nearest samples in the training data.  
    The predicted value is the average of the values of the k-nearest samples. 

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the dataset.
        :param dataset: Dataset object
        :return: The dataset
        """
        self.dataset = dataset
        return self

    def _get_closet_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        Calculates the mean of the class with the highest frequency.
        :param x: Array of samples.
        :return: Indexes of the classes with the highest frequency
        """

        # Calculates the distance between the samples and the dataset
        distances = self.distance(sample, self.dataset.X)

        # Sort the distances and get indexes
        k_nearest_neighbors = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # Calculate average
        return np.mean(k_nearest_neighbors_labels)


    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class with the highest frequency
        :return: Class with the highest frequency.
        """
        return np.apply_along_axis(self._get_closet_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the accuracy of the model.
        :return: Accuracy of the model.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)