import sys
sys.path.append(r'C:\Users\gonca\Documents\GitHub\Portefolio_SI\src')

import numpy as np
from typing import List, Tuple, Union
from models.decision_tree_classifier import DecisionTreeClassifier
from data.dataset import Dataset
from metrics.accuracy import accuracy


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_features: Union[int, None] = None,
                 min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        # set random seed
        if self.seed is not None:
            np.random.seed(self.seed)   

        # get number of samples and features
        n_samples, n_features = dataset.shape()

        # set max_features if not set
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))


        for i in range(self.n_estimators):
            # bootstrap dataset
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            features = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(dataset.X[bootstrap_indices, :][:, features], dataset.y[bootstrap_indices])
            # create tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_sample_split=self.min_sample_split, mode=self.mode)
            # fit tree
            tree.fit(bootstrap_dataset)
            # save features and tree as tuple
            self.trees.append((features, tree))
        return self
    
    
    def predict(self, dataset: Dataset) -> np.ndarray:

        y_pred = [None] * self.n_estimators
        for i, (features_idx, tree) in enumerate(self.trees):
            y_pred[i] = tree.predict(Dataset(dataset.X[:, features_idx], dataset.y))
        
        most_frequent = []
        for z in zip(*y_pred):
            most_frequent.append(max(set(z), key=z.count))

        return np.array(most_frequent)

    def score(self, dataset: Dataset) -> float:

        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
    
    if __name__ == '__main__':
        from iobla.csv_file import read_csv
        from model_selection.split import train_test_split
        from models.random_forest_classifier import RandomForestClassifier

        data = read_csv(r"C:\Users\gonca\Documents\GitHub\Portefolio_SI\datasets\iris\iris.csv", sep = ",", features = "True", label = True)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_sample_split=2, mode='gini', seed=42)
        rf_model.fit(train_data)
        test_score = rf_model.score(test_data)
        print(test_score)

        from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
        
        clf1 = skRandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=2)
        clf1.fit(train_data.X, train_data.y)
        y_pred = clf1.predict(test_data.X)
        score = accuracy(test_data.y, clf1.predict(test_data.X))
        print('Accuracy score sklearn: ', score)