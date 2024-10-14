from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.naive_bayes import GaussianNB as SKNB
from sklearn.ensemble import RandomForestClassifier as SKRF
from sklearn.linear_model import SGDClassifier as SGDSVM
from sklearn.tree import DecisionTreeClassifier as SKDT
import numpy as np


class LogisticRegression:
    def __init__(self, config):
        self.lr = SKLR(C=config['C'], max_iter=5000)

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.lr.fit(X_train, y_train)

    def predict_proba(self, X, with_logits=False):
        p = self.lr.predict_proba(X)
        if with_logits: return p, p
        else: return p
    
    def predict(self, X):
        return self.lr.predict(X)
    

class RandomPredictor:
    def __init__(self, config):
        pass

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        pass

    def predict_proba(self, X, with_logits=False):
        # return .5 for all classes deterministically
        p = np.ones((X.shape[0], 2)) * .5
        if with_logits: return p[:, 1], p
        else: return p[:, 1]
    
    def predict(self, X):
        return np.random.choice([0, 1], size=X.shape[0])


class NaiveBayes:
    def __init__(self, config):
        self.nb = SKNB()

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.nb.fit(X_train, y_train)

    def predict_proba(self, X, with_logits=False):
        p = self.nb.predict_proba(X)
        if with_logits: return p, p
        else: return p
    
    def predict(self, X):
        return self.nb.predict(X)


class RandomForest:
    def __init__(self, config):
        self.rf = SKRF(n_estimators=config['n_estimators'], 
                       max_depth=config['max_depth'],
                       min_samples_split=config['min_samples_split'])

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.rf.fit(X_train, y_train)

    def predict_proba(self, X, with_logits=False):
        p = self.rf.predict_proba(X)
        if with_logits: return p, p
        else: return p
    
    def predict(self, X):
        return self.rf.predict(X)
    

class SVM:
    def __init__(self, config):
        max_iter = 2000
        if 'max_iter' in config:
            max_iter = config['max_iter']
        self.svm = SGDSVM(loss='hinge', alpha=config['alpha'], max_iter=max_iter)

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.svm.fit(X_train, y_train)

    def predict_proba(self, X, with_logits=False):
        # Though scikit-learn offers a version of SVM
        # with predict_proba, this is really just Platt scaling
        # under the hood, and the base model is inefficient
        # on larger datasets. We opt for an SGD variant.
        p = self.svm.predict(X).astype(float)
        # expand to two classes
        p = np.vstack([1 - p, p]).T
        if with_logits: return p, p
        else: return p
    
    def predict(self, X):
        return self.svm.predict(X)


class DecisionTree:
    def __init__(self, config):
        self.dt = SKDT(max_depth=config['max_depth'],
                       min_samples_split=config['min_samples_split'])
        
    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.dt.fit(X_train, y_train)

    def predict_proba(self, X, with_logits=False):
        p = self.dt.predict_proba(X)
        if with_logits: return p, p
        else: return p
    
    def predict(self, X):
        return self.dt.predict(X)