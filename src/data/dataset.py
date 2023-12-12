from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self, subset = None) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    def dropna(self) -> 'Dataset':
        """
        Remove rows with NaN values from the dataset.

        This method removes rows containing NaN (Not a Number) values from the dataset.

        Returns
        -------
        Dataset
        Returns the updated Dataset object after removing rows with NaN values.
        """
        lines = np.isnan(self.X).any(axis=1)
        self.X = self.X[~lines]

        if self.has_label():
            self.y = self.y[~lines]
        return self
    
    def fillna(self, value) -> 'Dataset':
        """ 
        Fill the NaN (Not a Number) values in the dataset with a specified value.

        This method replaces all occurrences of NaN values in the features (self.X) with the specified fill value. 

        Parameters
        ----------
        value : str or float
            The value to fill NaN entries with. If `value` is a string, it can be one of the following options:
            - "mean": Fill NaN values with the mean of the non-NaN elements in the dataset.
            - "median": Fill NaN values with the median of the non-NaN elements in the dataset.
            If `value` is a float, fill NaN values with this specific numerical value.

        Returns
        -------
        Dataset
        Returns the updated Dataset object after filling NaN values.
        """
        if isinstance(value, str):
            if value == "mean":
                value_to_fill = np.nanmean(self.X)
            elif value == "median":
                value_to_fill = np.nanmedian(self.X)
            else:
                raise ValueError("Invalid value. Use a float or 'mean' or 'median'.")
        elif isinstance(value, (int, float)):
            value_to_fill = float(value)
        else:
            raise ValueError("Invalid value. Use a float or 'mean' or 'median'.")

        self.X[np.isnan(self.X)] = value_to_fill
        return self

    def remove_by_index(self, index) -> 'Dataset':
        """
        Remove a specific row from the dataset based on the given index.
        This method removes the row at the specified index from the features (self.X) and, if present, from the labels (self.y) of the dataset.

        Parameters
        ----------
        index : int
            The index of the row to be removed.

        Returns
        -------
        Dataset
        Returns the updated Dataset object after removing the specified row.
        """
        if index < 0 or index >= len(self.X):
            raise ValueError("Invalid index. Index out of range.")

        self.X = np.delete(self.X, index, axis=0)

        if self.has_label():
            self.y = np.delete(self.y, index, axis=0)
        return self

# Example usage:

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())