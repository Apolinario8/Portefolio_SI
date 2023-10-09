from typing import *
from dataclasses import dataclass
import numpy as np

def ValueRange(min, max):    #  ??
    pass

def train_test_split(dataset, random_state: int = 43, test_size: Annotated[float, ValueRange(0.0, 1.0)] = 0.3) -> Tuple:
    divisao = int(test_size * len(dataset.X))
    np.random_permutations(dataset)