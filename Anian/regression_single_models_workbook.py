import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.base import clone
from sklearn import tree
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, explained_variance_score)

pd.set_option("display.max_columns", None)

def preprocess_data(raw_data, target_variable):
    x_raw = raw_data.drop(target_variable, axis=1)
    y = raw_data[target_variable]
    categorical_features = x_raw.select_dtypes(include=['object']).columns.tolist()
    numerical_features = x_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
    binary_features = x_raw.select_dtypes(include=['bool']).columns.tolist()
    print("Categorical Features:", categorical_features)
    print("Numerical Features:", numerical_features)
    print("Binary Features:", binary_features)

    preprocess_pipeline = Pipeline([
        # Imputation (recode missing values)
        ('numerical_imputer', MeanMedianImputer(
            imputation_method='median',
            variables=numerical_features
        )),
        # ('categorical_imputer', CategoricalImputer(
        #     imputation_method='frequent',
        #     variables = categorical_features
        # )),
        # Handle outliers (cap outliers at boundary values)
        ('outlier_handler', Winsorizer(
            capping_method='gaussian',
            tail='both',
            fold='auto',
            variables=numerical_features
        )),
        # Normalize numeric data
        ('scaler', SklearnTransformerWrapper(
            transformer=StandardScaler(),
            variables=numerical_features
        ))  # ,
        # Encode categorical data (create dummy variables if there are categorical variables)
        # ('encoder', OneHotEncoder(
        #     variables= categorical_features,
        #     drop_last=True
        # ))
    ])

    # Pre-process raw_data
    x_processed = preprocess_pipeline.fit_transform(x_raw)
    return x_processed, y

def search_model(param_distribution, learning_model, cross_validation=5, iteration=10, random_seed=33,
                 search_method='random'):
    if learning_model == 'tree':
        regressor = tree.DecisionTreeRegressor(random_state=random_seed)
    elif learning_model == 'ridge':
        regressor = Ridge(random_state=random_seed)
    elif learning_model == 'lasso':
        regressor = Lasso(random_state=random_seed)
    elif learning_model == 'elasticnet':
        regressor = ElasticNet(random_state=random_seed)
    elif learning_model == 'svr':
        regressor = SVR()
    elif learning_model == 'nn':
        regressor = MLPRegressor(random_state=random_seed)
    else:
        return None

    if search_method == 'grid':
        pre_train_model = GridSearchCV(
            estimator=regressor,
            param_grid=param_distribution,
            cv=cross_validation,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2
        )
    else:  # always perform a random search if the user enters anything other than 'grid'
        search_method = 'random'
        pre_train_model = RandomizedSearchCV(
            estimator=regressor,
            param_distributions=param_distribution,
            n_iter=iteration,  # Number of parameter settings sampled
            cv=cross_validation,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            random_state=random_seed
        )
    return pre_train_model

def train_tree_model(x_train, y_train, random_seed=33, search_method='random'):
    param_space = {
        'max_depth': [None] + list(range(3, 31)),
        'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 5, 7, 10],
        'max_features': [None, 'sqrt', 'log2'] + list(np.arange(0.5, 1.0, 0.1)),
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    }

    model = search_model(param_distribution=param_space, learning_model="tree", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def explain_tree_model(tree_model, x_train):
    # Decision Tree
    feature_names = x_train.columns.tolist()
    plt.figure(figsize=(20, 10))
    tree.plot_tree(tree_model,
                   feature_names=feature_names,
                   filled=True,
                   rounded=True,
                   fontsize=8)
    plt.title("Decision Tree Visualization")

    # Feature Importance
    plt.figure(figsize=(20, 10))
    feature_importance = tree_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names,
                                  'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
    cmap = plt.cm.Greens
    norm = plt.Normalize(importance_df['Importance'].min(), importance_df['Importance'].max())
    colors = cmap(norm(importance_df['Importance']))

    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance - Decision Tree")
    plt.show()

def train_ridge_model(x_train, y_train, random_seed=33, search_method='random'):
    if search_method == 'grid':
        param_space = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'fit_intercept': [True, False]
        }
    else:
        search_method = 'random'
        param_space = {
            'alpha': loguniform(0.001, 100),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'fit_intercept': [True, False]
        }

    model = search_model(param_distribution=param_space, learning_model="ridge", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def train_lasso_model(x_train, y_train, random_seed=33, search_method='random'):
    if search_method == 'grid':
        param_space = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random'],
            'max_iter': [1000, 2000, 5000]
        }
    else:
        search_method = 'random'
        param_space = {
            'alpha': loguniform(0.001, 100),
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random'],
            'max_iter': [1000, 2000, 5000]
        }
    model = search_model(param_distribution=param_space, learning_model="lasso", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def train_elasticnet_model(x_train, y_train, random_seed=33, search_method='random'):
    if search_method == 'grid':
        param_space = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000]
        }
    else:
        search_method = 'random'
        param_space = {
            'alpha': loguniform(0.001, 10),
            'l1_ratio': uniform(0, 1),
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 5000]
        }
    model = search_model(param_distribution=param_space, learning_model="elasticnet", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def explain_regularized_model(model, x_train):
    # Feature coefficients
    feature_names = x_train.columns.tolist()
    coefficients = model.coef_

    # Create DataFrame for visualization
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(12, 8))
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-abs(coefficients).max(), abs(coefficients).max())
    colors = cmap(norm(coef_df['Coefficient']))

    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Coefficient Value')
    plt.title(f'{type(model).__name__} Coefficients')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

    # Print intercept and regularization strength
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Alpha (regularization strength): {model.alpha:.6f}")

    # For ElasticNet, also show l1_ratio
    if hasattr(model, 'l1_ratio'):
        print(f"L1 ratio: {model.l1_ratio:.4f}")

    # Count non-zero coefficients (feature selection effect)
    non_zero = np.sum(coefficients != 0)
    print(f"Non-zero coefficients: {non_zero} out of {len(coefficients)} ({non_zero / len(coefficients) * 100:.1f}%)")

def train_svm_model(x_train, y_train, random_seed=33, search_method='random'):
    if search_method == 'grid':
        param_space = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'epsilon': [0.01, 0.1, 0.2]
        }
    else:
        search_method = 'random'
        param_space = {
            'C': loguniform(0.1, 1000),
            'gamma': loguniform(0.0001, 1),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'epsilon': loguniform(0.01, 1)
        }

    model = search_model(param_distribution=param_space, learning_model="svr", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def train_nn_model(x_train, y_train, random_seed=33, search_method='random'):
    if search_method == 'grid':
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000],
            'batch_size': [32, 64, 128]
        }
    else:
        search_method = 'random'
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100),
                                   (200, 100), (100, 50, 25), (200, 100, 50)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': loguniform(1e-5, 1),
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
            'learning_rate_init': loguniform(0.0001, 0.1),
            'max_iter': randint(100, 1000),
            'batch_size': randint(16, 256),
            'early_stopping': [True],
            'n_iter_no_change': randint(5, 30)
        }

    model = search_model(param_distribution=param_space, learning_model="nn", search_method=search_method)
    model.fit(x_train, y_train)
    print(f"Best hyperparameters found by {search_method} search:")
    print(model.best_params_)
    print(f"Best neg MSE score: {model.best_score_:.4f}")
    return model.best_estimator_

def explain_nn_model(nn_model):
    # Training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(nn_model.loss_curve_, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.show()

    # Weights of the first hidden layer
    weights = nn_model.coefs_[0]  # Access weights of the first hidden layer
    plt.figure(figsize=(12, 8))
    plt.matshow(weights, cmap='viridis', fignum=1)
    plt.colorbar()
    plt.title("Weight Matrix of First Hidden Layer")
    plt.xlabel("Neurons in First Hidden Layer")
    plt.ylabel("Input Features")
    plt.show()

def explain_feature_importance(trained_model, x_train, y_train, x_test, y_test):
    feature_names = x_train.columns.tolist()

    # Permutation importance
    result = permutation_importance(
        trained_model, x_test, y_test,
        n_repeats=10,
        random_state=33,
        scoring='neg_mean_squared_error'
    )
    importance = -result.importances_mean  # Negate because we want higher MSE to mean more important
    std = result.importances_std

    plt.figure(figsize=(10, 8))
    sorted_idx = importance.argsort()
    plt.barh(
        range(len(sorted_idx)),
        importance[sorted_idx],
        xerr=std[sorted_idx],
        align='center'
    )
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Increase in MSE when feature is permuted')
    plt.title('Feature Importance (Permutation Method)')
    plt.text(
        x=max(importance) * 0.6,
        y=len(sorted_idx) * 0.1,
        s="Error bars show standard deviation\nacross 10 permutation runs",
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5},
        fontsize=9
    )
    plt.tight_layout()
    plt.show()

    # Leave-one-feature-out
    base_score = mean_squared_error(y_test, trained_model.predict(x_test))
    feature_importance = {}
    for feature in x_train.columns:
        x_train_dropped = x_train.drop(columns=[feature])
        x_test_dropped = x_test.drop(columns=[feature])
        model_dropped = clone(trained_model)
        model_dropped.fit(x_train_dropped, y_train)
        # Calculate MSE on the modified test set
        mse_dropped = mean_squared_error(y_test, model_dropped.predict(x_test_dropped))
        # Importance is the increase in error when feature is removed
        feature_importance[feature] = mse_dropped - base_score
        print(feature, feature_importance[feature])

    importance_df = pd.DataFrame(list(feature_importance.items()),
                                 columns=['Feature', 'Importance']).sort_values(
        by="Importance", ascending=False)
    cmap = plt.cm.Reds
    norm = plt.Normalize(importance_df['Importance'].min(),
                         importance_df['Importance'].max())
    colors = cmap(norm(importance_df['Importance']))
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Increase in MSE (LOFO)")
    plt.title("Feature Importance (Leave-One-Feature-Out)")
    plt.tight_layout()
    plt.show()

def assess_model_performance(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)

    # Print metrics
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

def compare_regression_models(models, x_test, y_test):
    results = {
        'Model': [],
        'R²': [],
        'Adjusted R²': [],
        'MSE': [],
        'RMSE': [],
        'MAE': [],
        'Explained Variance': []
    }
    all_predictions = {}
    n_samples = x_test.shape[0]
    n_features = x_test.shape[1]

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating {name}")
        print(f"{'=' * 50}")

        y_pred = model.predict(x_test)
        all_predictions[name] = y_pred

        r2 = r2_score(y_test, y_pred)
        # Adjusted R² formula: 1 - (1 - R²) * (n - 1) / (n - p - 1)
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)

        results['Model'].append(name)
        results['R²'].append(r2)
        results['Adjusted R²'].append(adj_r2)
        results['MSE'].append(mse)
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['Explained Variance'].append(explained_var)

    results_df = pd.DataFrame(results)

    metrics_to_plot = ['R²', 'Adjusted R²', 'Explained Variance']

    plt.figure(figsize=(15, 8))

    bar_width = 0.25
    x = np.arange(len(results_df['Model']))

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * bar_width, results_df[metric], width=bar_width, label=metric)

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison - Goodness of Fit')
    plt.xticks(x + bar_width, results_df['Model'])
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    error_metrics = ['MSE', 'RMSE', 'MAE']
    plt.figure(figsize=(15, 8))

    for i, metric in enumerate(error_metrics):
        plt.subplot(1, 3, i + 1)
        plt.bar(results_df['Model'], results_df[metric], color='skyblue')
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))

    n_models = len(models)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    for i, (name, y_pred) in enumerate(all_predictions.items(), 1):
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(y_test, y_pred, alpha=0.5)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{name} - Actual vs Predicted')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    metrics_df = results_df.drop('Model', axis=1)
    metrics_df.index = results_df['Model']

    for col in ['MSE', 'RMSE', 'MAE']:
        max_val = metrics_df[col].max()
        metrics_df[col] = 1 - (metrics_df[col] / max_val)

    sns.heatmap(metrics_df, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title('Model Performance Metrics Comparison\n(Higher values are better for all metrics in this visualization)')
    plt.tight_layout()
    plt.show()

    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    print("\nBest Models by Metric:")
    for metric in ['R²', 'Adjusted R²', 'Explained Variance']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"Best {metric}: {best_model} ({best_score:.4f})")

    for metric in ['MSE', 'RMSE', 'MAE']:
        best_idx = results_df[metric].idxmin()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, metric]
        print(f"Best {metric}: {best_model} ({best_score:.4f})")

    return results_df
