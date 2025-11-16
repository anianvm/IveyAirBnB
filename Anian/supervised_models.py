import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.neural_network import MLPRegressor

from sklearn.inspection import permutation_importance


# ===========================================================================
# 1. PREPARE REGRESSION DATA
# ===========================================================================
def prepare_regression_data(df_model, target_variable="price"):
    """
    Flexible target variable.
    Uses the scaled target already inside df_model.
    """

    if target_variable not in df_model.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in df_model.")

    # y = scaled target
    y = df_model[target_variable].copy()

    # X = everything except target
    X = df_model.drop(columns=[target_variable], errors="ignore")

    return X, y



# ===========================================================================
# 2. METRICS
# ===========================================================================
def evaluate_model(model, X_test, y_test):
    """
    Return MAE, RMSE, R2, and Adjusted R2.
    """
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return mae, rmse, r2, adj_r2



# ===========================================================================
# 3. FEATURE IMPORTANCE
# ===========================================================================
def get_feature_importance(model, X_train):
    """
    Handle importance for:
    - Linear Regression (coef)
    - Tree/Forest/GBM (feature_importances_)
    - MLP (permutation importance)
    """

    if hasattr(model, "coef_"):
        # Linear regression
        return pd.Series(model.coef_, index=X_train.columns)

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        return pd.Series(model.feature_importances_, index=X_train.columns)

    # Default = permutation importance (MLP)
    perm = permutation_importance(model, X_train, model.predict(X_train), n_repeats=5)
    return pd.Series(perm.importances_mean, index=X_train.columns)



# ===========================================================================
# 4. TRAIN SINGLE MODELS (CBS/Ivey Style)
# ===========================================================================
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_mlp_regressor(X_train, y_train):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model



# ===========================================================================
# 5. TRAIN ENSEMBLE MODELS (Voting + Stacking)
# ===========================================================================
def train_voting_ensemble(X_train, y_train):
    model = VotingRegressor([
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=200)),
        ("gb", GradientBoostingRegressor())
    ])
    model.fit(X_train, y_train)
    return model


def train_stacking_ensemble(X_train, y_train):
    model = StackingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("dt", DecisionTreeRegressor()),
            ("rf", RandomForestRegressor(n_estimators=200))
        ],
        final_estimator=GradientBoostingRegressor()
    )
    model.fit(X_train, y_train)
    return model



# ===========================================================================
# 6. TRAIN ALL MODELS TOGETHER
# ===========================================================================
def train_all_models(
        X,
        y,
        test_size=0.2,
        random_state=42
    ):
    """
    Main controller function (like insurance notebooks).
    Produces:
    - performance table
    - feature importance dict
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # List of models to evaluate
    model_functions = {
        "Linear Regression": train_linear_regression,
        "Decision Tree": train_decision_tree,
        "Random Forest (Bagging)": train_random_forest,
        "Gradient Boosting (Boosting)": train_gradient_boosting,
        "Neural Network (MLP)": train_mlp_regressor,
        "Voting Ensemble": train_voting_ensemble,
        "Stacking Ensemble": train_stacking_ensemble
    }

    performance = []
    importances = {}

    # Loop through each model type
    for name, func in model_functions.items():

        model = func(X_train, y_train)

        mae, rmse, r2, adj_r2 = evaluate_model(model, X_test, y_test)
        performance.append([name, mae, rmse, r2, adj_r2])

        importances[name] = get_feature_importance(model, X_train)

    # Summary DataFrame
    results_df = pd.DataFrame(
        performance,
        columns=["Model", "MAE", "RMSE", "R²", "Adjusted R²"]
    )

    return results_df, importances
