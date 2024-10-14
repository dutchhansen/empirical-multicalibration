from sklearn.isotonic import IsotonicRegression as Isotonic

class IsotonicRegression:
    """
    Temperature scaling method for calibration.
    """
    def __init__(self, params):
        self.ir = Isotonic(out_of_bounds="clip")

    def fit(self, confs, labels, subgroups):
        self.ir.fit(confs, labels)

    def batch_predict(self, f_xs, groups):
        return self.ir.predict(f_xs)