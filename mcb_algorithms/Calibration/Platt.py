from sklearn.calibration import _SigmoidCalibration as PlattCalibration
import numpy as np

class PlattScaling:
    """
    Platt scaling, also referred to as the
    `sigmoid` or `logistic` calibration method.
    Fits logistic regression to model predictions.
    """
    def __init__(self, params):
        self.pc = PlattCalibration()

    def fit(self, confs, labels, subgroups):
        self.pc.fit(confs, labels)

    def batch_predict(self, f_xs, groups):
        return self.pc.predict(f_xs)
