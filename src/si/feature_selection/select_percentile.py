from si.data.dataset import Dataset

class SelectPercentile:

    def __init__(self, score_funct, percentile):
        self.score_funct = score_funct
        self.percentile = percentile
        self.F = None
        self.P = None 

    def fit(self):
        