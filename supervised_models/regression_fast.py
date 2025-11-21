import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import loguniform, randint, uniform
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor


# ---- FAST SEARCH WRAPPER -----

# We'll make this more robust: 20 iterations and 5-fold CV.
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
    # teacher's `train_tree_model` param_space
    param_dist = {
        'max_depth': [None] + list(range(3, 31)),
        'min_samples_split': randint(2, 21),
        'min_samples_leaf': randint(1, 11),
        'max_features': [None, 'sqrt', 'log2'] + list(np.arange(0.5, 1.0, 0.1)),
        'criterion': ['squared_error', 'friedman_mse', 'poisson']
    }
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

def train_svr_fast(X, y):
    # Based on teacher's train_svm_model
    # We focus on 'rbf' as 'linear' is covered by LinearSVR
    param_dist = {
        'C': loguniform(0.1, 1000),
        'gamma': loguniform(0.0001, 1),
        'kernel': ['rbf', 'poly'], # 'rbf' is the most powerful
        'epsilon': loguniform(0.01, 1)
    }
    # Note: SVR can be slow, so n_iter=10 might be safer here
    # return fast_search(SVR(), param_dist, X, y, n_iter=10)
    return fast_search(SVR(), param_dist, X, y, cv=3, n_iter=10)

def train_adaboost_fast(X, y):
    # Based on your teacher's adp_boost
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 1.0),
        'loss': ['linear', 'square', 'exponential']
    }
    return fast_search(AdaBoostRegressor(random_state=42), param_dist, X, y)

def train_mlp_fast(X, y):
    # teacher's train_nn_model
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'], # Adam is the modern default
        'alpha': loguniform(1e-5, 1.0),
        'learning_rate_init': loguniform(0.0001, 0.01),
        'early_stopping': [True],
        'n_iter_no_change': [10]
    }
    # Use max_iter=1000 and let early_stopping find the right time
    return fast_search(MLPRegressor(random_state=42, max_iter=1000), param_dist, X, y)

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
