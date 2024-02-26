import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

class TargetEstimator(BaseEstimator):
    """ Define the target estimator on latent space network features.
    """
    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.target_estimator = GridSearchCV(
            Ridge(), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2", n_jobs=n_jobs)

    def fit(self, X, y, scoring = None):
        score = np.nan
        self.target_estimator.fit(X, y)
        if scoring is not None:
            score = self.score(X, y, scoring)
        return score

    def predict(self, X):
        y_pred = self.target_estimator.predict(X)
        return y_pred
    
    def score(self, X, y, scoring):
        y_pred = self.target_estimator.predict(X)
        if scoring == 'mape':
            score = mean_absolute_percentage_error(y, y_pred)
        elif scoring == 'mae':
            score = mean_absolute_error(y, y_pred)
        elif scoring == 'r2':
            score = r2_score(y, y_pred)
        return score
    
class SiteEstimator(BaseEstimator):
    """ Define the site estimator on latent space network features.
    """
    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.site_estimator = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=150), cv=5,
            param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy", n_jobs=n_jobs)

    def fit(self, X, y):
        self.site_estimator.fit(X, y)
        return self.site_estimator.score(X, y)

    def predict(self, X):
        return self.site_estimator.predict(X)

    def score(self, X, y):
        return self.site_estimator.score(X, y)