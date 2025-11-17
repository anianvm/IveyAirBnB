# regression_single_models_fast.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform, randint

# -------------------------
# FAST SEARCH WRAPPER
# -------------------------
def fast_search(model, param_dist, X, y, cv=3, n_iter=10):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        scoring="neg_mean_squared_error",
        random_state=42
    )
    search.fit(X, y)
    print("Best Params:", search.best_params_)
    print("Best Score:", search.best_score_)
    return search.best_estimator_


# -------------------------
# FAST MODELS
# -------------------------

def train_tree_fast(X, y):
    param_dist = {
        "max_depth": randint(3, 25),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10)
    }
    return fast_search(DecisionTreeRegressor(), param_dist, X, y)


def train_forest_fast(X, y):
    param_dist = {
        "n_estimators": randint(50, 150),
        "max_depth": randint(5, 20),
        "min_samples_leaf": randint(1, 5)
    }
    return fast_search(RandomForestRegressor(n_jobs=-1), param_dist, X, y)


def train_gboost_fast(X, y):
    param_dist = {
        "n_estimators": randint(50, 150),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(2, 6)
    }
    return fast_search(GradientBoostingRegressor(), param_dist, X, y)


def train_ridge_fast(X, y):
    param_dist = {"alpha": loguniform(0.01, 10)}
    return fast_search(Ridge(), param_dist, X, y)


def train_lasso_fast(X, y):
    param_dist = {"alpha": loguniform(0.001, 1)}
    return fast_search(Lasso(max_iter=5000), param_dist, X, y)


def train_elastic_fast(X, y):
    param_dist = {
        "alpha": loguniform(0.001, 1),
        "l1_ratio": loguniform(0.01, 0.99)
    }
    return fast_search(ElasticNet(max_iter=5000), param_dist, X, y)


def train_linear_svr_fast(X, y):
    param_dist = {
        "C": loguniform(0.01, 10),
        "epsilon": loguniform(0.001, 1)
    }
    return fast_search(LinearSVR(max_iter=5000), param_dist, X, y)

# regression_ensemble_fast.py
from sklearn.ensemble import (
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.neural_network import MLPRegressor


def bag_fast(models, X_train, y_train):
    bag = BaggingRegressor(
        estimators=list(models.items()),
        n_estimators=10,
        n_jobs=-1,
        random_state=42
    )
    bag.fit(X_train, y_train)
    return bag


def vote_fast(models):
    return VotingRegressor(list(models.items()))


def stack_fast(models):
    meta = MLPRegressor(hidden_layer_sizes=(20,), max_iter=300, random_state=42)
    return StackingRegressor(
        estimators=list(models.items()),
        final_estimator=meta,
        n_jobs=-1
    )
