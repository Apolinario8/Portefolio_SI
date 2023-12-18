import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) using Singular Value Decomposition (SVD).
    
    Parameters
    ----------
    n_components : int
        Number of components to keep.

    Attributes
    ----------
    mean : np.ndarray
        Mean of the samples.
    components : np.ndarray
        Principal components (unitary matrix of eigenvectors).
    explained_variance : np.ndarray
        Explained variance (diagonal matrix of eigenvalues).

    Methods
    -------
    fit(X)
        Estimate the mean, principal components, and explained variance from the input data.
    transform(X)
        Reduce the input dataset to its principal components.
    fit_transform(X)
        Run fit on the input data and then transform it.

    Notes
    -----
    PCA is a linear algebra technique used for dimensionality reduction.
    It projects data into a lower-dimensional space while maximizing variance.

    PCA using SVD involves the following steps:
    1. Center the data: Subtract the mean from the dataset.
    2. Calculate SVD: Decompose the centered data matrix using Singular Value Decomposition.
    3. Infer Principal Components: Extract the first n_components of right singular vectors.
    4. Infer Explained Variance: Compute eigenvalues from singular values for explained variance.

    The transformed data can be calculated using the dot product of centered data and principal components.

    Examples
    --------
    # Create PCA object with 2 components
    pca = PCA(n_components=2)
    
    # Fit and transform the data
    reduced_data = pca.fit_transform(data)
    """

    def __init__(self, n_components: int):
        """
        Initialize PCA with the specified number of components.
        
        Parameters
        ----------
        n_components : int
            Number of components to keep.
        """
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Estimate the mean, principal components, and explained variance from the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        PCA
            The PCA object.
        """
        self.mean = np.mean(X, axis=0)
        centered_X = X - self.mean
        
        U, S, VT = np.linalg.svd(centered_X, full_matrices=False)
        
        self.components = VT[:self.n_components]

        for i in range(self.n_components):   #making sure the components have the same sign as the sklearn implementation (first element of each component is positive) --> not necessary
            if np.sum(self.components[i]) < 0:
                self.components[i] = -self.components[i]
        
        self.explained_variance = (S[:self.n_components] ** 2) / (X.shape[0] - 1)

        total_variance = np.sum(S ** 2)
        self.explained_variance_ratio = (S[:self.n_components] ** 2) / total_variance
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the input dataset to its principal components.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Reduced dataset, shape (n_samples, n_components).
        """
        centered_X = X - self.mean
        
        X_reduced = np.dot(centered_X, self.components.T) 

        return X_reduced
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Run fit on the input data and then transform it.

        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Reduced dataset, shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)