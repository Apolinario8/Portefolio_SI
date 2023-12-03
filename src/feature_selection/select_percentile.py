from typing import Callable
import numpy as np
import sys
sys.path.append(r'C:\Users\gonca\Documents\GitHub\Portefolio_SI\src')
from data.dataset import Dataset
from statisticsbla.f_classification import f_classification

class SelectPercentile:
    """
    Select features according to a percentile of the highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: int, default=10
        Percentile of features to select.
        
    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    threshold: float
        The threshold value used for feature selection based on the percentile.
    """
    def __init__(self, score_func: Callable = f_classification, percentile: int = 10):
        """
        Select features according to a percentile of the highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=10
            Percentile of features to select.
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None
        self.threshold = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        self.threshold = np.percentile(self.F, 100 - self.percentile)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting features above the threshold.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with features selected based on the percentile threshold.
        """
        above_threshold = self.F > self.threshold
        features = np.array(dataset.features)[above_threshold]
        return Dataset(X=dataset.X[:, above_threshold], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting features above the threshold.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with features selected based on the percentile threshold.
        """
        self.fit(dataset)
        return self.transform(dataset)