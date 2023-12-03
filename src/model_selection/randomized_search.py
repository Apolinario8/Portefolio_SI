from typing import Dict, Tuple, Callable, Union

import numpy as np

from data.dataset import Dataset
from model_selection.cross_validate import cross_validate

Num = Union[int, float]


def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         test_size: float = 0.3) -> Dict[str, Tuple[str, Num]]:
    scores = {
        'parameters': [],
        'seed': [],
        'train': [],
        'test': []
    }

    #  checks if parameters exist in the model
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The {model} does not have parameter {parameter}.")

    #  sets n_iter parameter combinations
    for i in range(n_iter):

        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(random_state)

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in parameter_distribution.items():
            # set the combination of parameter and its values to the model
            parameters[parameter] = np.random.choice(value)  # choose a random value from the distribution

        # set the parameters to the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # performs cross_validation with the combination
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # stores the parameter combination and the obtained scores
        scores['parameters'].append(parameters)
        scores['train'].append(score['train'])
        scores['test'].append(score['test'])

    return scores