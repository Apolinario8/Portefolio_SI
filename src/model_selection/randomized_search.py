import sys
sys.path.append(r'C:\Users\gonca\Documents\GitHub\Portefolio_SI\src')

from typing import Dict, Tuple, Callable, Union
import numpy as np
from data.dataset import Dataset
from model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, n_iter=None, random_state=None):
   
    for parameter in hyperparameter_grid:
            if not hasattr(model, parameter):
                raise AttributeError(f"Model {model} does not have parameter {parameter}.") 

    results = {'scores': [], 'hyperparameters': []}

    for _ in range(n_iter):
        parameters = {}
        for key, values in hyperparameter_grid.items():
            # Choose a different random value for each hyperparameter
            parameters[key] = np.random.choice(values)

        # Set the hyperparameters in the model
        for key, value in parameters.items():
            setattr(model, key, value)

        # Cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Record the score and hyperparameters for this iteration
        results['scores'].append(np.mean(score))
        results['hyperparameters'].append(parameters)

    # Identify the best score and best hyperparameters
    best_index = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_index]
    results['best_score'] = np.max(results['scores'])

    return results