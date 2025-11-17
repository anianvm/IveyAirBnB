import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform, randint, uniform

# -------------------------
# FAST SEARCH WRAPPER
# -------------------------
# We'll make this more robust: 20 iterations and 5-fold CV.
# This is still MUCH faster than a full GridSearch.
def fast_search(model, param_dist, X, y, cv=5, n_iter=20):
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

# In regression_fast.py, for train_tree_fast
def train_tree_fast(X, y):
    # This is based on your teacher's `train_tree_model` param_space
    param_dist = {
        'max_depth': [None] + list(range(3, 31)),
        'min_samples_split': randint(2, 21), # randint(2, 20) is also fine
        'min_samples_leaf': randint(1, 11),
        'max_features': [None, 'sqrt', 'log2'] + list(np.arange(0.5, 1.0, 0.1)),
        'criterion': ['squared_error', 'friedman_mse', 'poisson']
    }
    # Added random_state=42
    return fast_search(DecisionTreeRegressor(random_state=42), param_dist, X, y)


def train_forest_fast(X, y):
    param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": randint(5, 25),
        "min_samples_leaf": randint(1, 6),
        'min_samples_split': randint(2, 11),
        'max_features': [None, 'sqrt', 'log2', 0.5, 0.7, 0.9]
    }
    return fast_search(RandomForestRegressor(n_jobs=-1, random_state=42), param_dist, X, y)


def train_gboost_fast(X, y):
    param_dist = {
        "n_estimators": randint(50, 200),
        "learning_rate": loguniform(0.01, 0.2),
        "max_depth": randint(2, 8),
        "min_samples_leaf": randint(1, 10),
        "subsample": uniform(0.6, 0.4)
    }
    return fast_search(GradientBoostingRegressor(random_state=42), param_dist, X, y)


def train_ridge_fast(X, y):
    # based on teachers params
    param_dist = {
        'alpha': loguniform(0.001, 100),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'fit_intercept': [True, False]
    }
    return fast_search(Ridge(random_state=42), param_dist, X, y)


def train_lasso_fast(X, y):
    # based on teachers params
    param_dist = {
        'alpha': loguniform(0.001, 100),
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random'],
        'max_iter': [1000, 2000, 5000]
    }
    return fast_search(Lasso(max_iter=5000, random_state=42), param_dist, X, y)


def train_elastic_fast(X, y):
    # teachers params
    param_dist = {
        'alpha': loguniform(0.001, 10),
        'l1_ratio': uniform(0, 1), # Use uniform(0,1) instead of loguniform
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000, 5000] # Kept your high max_iter
    }
    return fast_search(ElasticNet(max_iter=5000, random_state=42), param_dist, X, y)


def train_linear_svr_fast(X, y):
    param_dist = {
        "C": loguniform(0.01, 10),
        "epsilon": loguniform(0.001, 1),
        "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        "fit_intercept": [True, False]
    }
    return fast_search(LinearSVR(max_iter=5000, random_state=42, dual="auto"), param_dist, X, y)

# regression_ensemble_fast.py
from sklearn.ensemble import (
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.neural_network import MLPRegressor


def bag_fast(models, X_train, y_train):
    # auto-choose best model (lowest MAE on training set)
    errors = {name: mean_squared_error(y_train, model.predict(X_train))
              for name, model in models.items()}
    best = min(errors, key=errors.get)

    bag = BaggingRegressor(
        estimator=models[best],
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
