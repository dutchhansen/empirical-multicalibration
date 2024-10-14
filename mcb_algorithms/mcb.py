import numpy as np
from tqdm import trange
from mcb_algorithms.HKRR.hkrr import HKRRAlgorithm
from mcb_algorithms.HJZ.hjz import HJZAlgorithm
from mcb_algorithms.Calibration.Platt import PlattScaling
from mcb_algorithms.Calibration.Temperature import TemperatureScaling
from mcb_algorithms.Calibration.Isotonic import IsotonicRegression


## MCB algorithm
class MulticalibrationPredictor:
    """
    General Multicalibration Predictor class.
    """
    
    def __init__(self, algorithm, params):
        """
        Initialize Multicalibration Predictor.
        """
        self.algorithm = algorithm
        self.params = params
        if algorithm == 'HKRR':
            self.mcbp = HKRRAlgorithm(params)
        elif algorithm == 'HJZ':
            self.mcbp = HJZAlgorithm(params)
        elif algorithm == 'Platt':
            self.mcbp = PlattScaling(params)
        elif algorithm == 'Temp':
            self.mcbp = TemperatureScaling(params)
        elif algorithm == 'Isotonic':
            self.mcbp = IsotonicRegression(params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")

    def fit(self, confs, labels, subgroups):
        """
        Returns vector of confidences on calibration set.

        HKRR: alpha, lmbda, use_oracle=True, randomized=True, max_iter=float('inf')
        """
        self.mcbp.fit(confs, labels, subgroups)

    def batch_predict(self, f_xs, groups):
        """
        Returns calibrated predictions for a batch of data points.
        HKRR: early_stop=None
        """
        return self.mcbp.batch_predict(f_xs, groups)