from typing import Tuple
import numpy as np
from data.dataset import Dataset

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Stratified split of the dataset into training and testing sets.
    
    Parameters
    ----------
    dataset: Dataset
        The Dataset object to split into training and testing data.
    test_size: float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
    random_state: int, optional (default=42)
        The seed for generating permutations.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the stratified train and test Dataset objects.
    """
    np.random.seed(random_state)
    
    # Get unique class labels and their counts
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)
    
    # Initialize empty lists for train and test indices
    train_indices = []
    test_indices = []
    
    # Loop through unique labels
    for label in unique_labels:
        # Calculate the number of test samples for the current class
        num_test_samples = int(label_counts[label] * test_size)
        
        # Shuffle and select indices for the current class
        class_indices = np.where(dataset.y == label)[0]
        np.random.shuffle(class_indices)
        
        # Add indices to the test set
        test_indices.extend(class_indices[:num_test_samples])
        
        # Add remaining indices to the train set
        train_indices.extend(class_indices[num_test_samples:])
    
    # Create training and testing datasets
    train_data = Dataset(dataset.X[train_indices], dataset.y[train_indices],
                         features=dataset.features, label=dataset.label)
    test_data = Dataset(dataset.X[test_indices], dataset.y[test_indices],
                        features=dataset.features, label=dataset.label)
    
    return train_data, test_data