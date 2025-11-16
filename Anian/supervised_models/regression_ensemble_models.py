import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (BaggingRegressor, VotingRegressor, StackingRegressor, RandomForestRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score)
import regression_single_models as rsmw

def bag_models(models, x_train, y_train, x_test, y_test):
    bagging_models = {}
    bagging_results = {}

    for name, model in models.items():
        bagging = BaggingRegressor(estimator=model, n_estimators=20, random_state=42)
        bagging.fit(x_train, y_train)
        preds = bagging.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        bagging_results[name] = r2
        bagging_models[name] = bagging
        print(f"Bagging with {name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    best_model_name = max(bagging_results, key=bagging_results.get)
    best_bagging_model = bagging_models[best_model_name]
    return best_model_name, best_bagging_model

def vote_models(models):
    models_tuple = [(name, model) for name, model in models.items()]
    voting_model = VotingRegressor(estimators=models_tuple)
    return voting_model

def stack_models(models, meta_learner):
    models_tuple = [(name, model) for name, model in models.items()]
    stacking_model = StackingRegressor(estimators=models_tuple, final_estimator=meta_learner)
    return stacking_model

def adp_boost(models, x_train, y_train, x_test, y_test):
    boosting_models = {}
    boosting_results = {}

    for name, model in models.items():
        try:
            boosting = AdaBoostRegressor(estimator=model, n_estimators=50, random_state=42)
            boosting.fit(x_train, y_train)
            preds = boosting.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            boosting_models[name] = boosting
            boosting_results[name] = r2
            print(f"AdaBoost with {name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            print(f"AdaBoost could not be applied to {name}: {str(e)}")
    if boosting_results:
        best_model_name = max(boosting_results, key=boosting_results.get)
        best_boosting_model = boosting_models[best_model_name]
        return best_model_name, best_boosting_model

def assess_regression_performance(model, x_test, y_test):
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Explained Variance Score: {ev:.4f}")

    # Residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)

    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2)

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Return metrics for comparison
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Explained Variance': ev
    }
