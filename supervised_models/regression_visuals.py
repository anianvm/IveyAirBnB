import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Tree explanation
def explain_tree_model(model, feature_names):
    """
    Plots simple feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_. Skipping tree explain.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[idx][:15], y=np.array(feature_names)[idx][:15])
    plt.title("Top 15 Feature Importances (Tree-Based Model)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# reg. coefficient plot
def explain_regularized_model(model, feature_names):
    if not hasattr(model, "coef_"):
        print("Model has no coef_. Skipping regularized explain.")
        return

    coefs = model.coef_
    idx = np.argsort(np.abs(coefs))[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=coefs[idx][:15],
        y=np.array(feature_names)[idx][:15],
    )
    plt.title("Top 15 Coefficients (Regularized Model)")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# permutation importance
def explain_feature_importance(model, X_train, y_train, X_test, y_test, n_repeats=5):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    idx = np.argsort(result.importances_mean)[::-1]

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=result.importances_mean[idx][:15],
        y=X_train.columns[idx][:15]
    )
    plt.title("Top 15 Permutation Importances")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# model performance table
def assess_model_performance(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

# performance bar plots
def compare_regression_models(model_dict, X_test, y_test):
    results = {}

    for name, model in model_dict.items():
        preds = model.predict(X_test)
        results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        }

    df_res = pd.DataFrame(results).T
    print("\nMODEL PERFORMANCE:\n", df_res)

    #  Barplot: R2
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df_res.index, y=df_res["R2"])
    plt.title("R² Score Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #  Barplot: RMSE
    plt.figure(figsize=(8, 4))
    sns.barplot(x=df_res.index, y=df_res["RMSE"])
    plt.title("RMSE Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #  Scatter: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    for name, model in model_dict.items():
        preds = model.predict(X_test)
        plt.scatter(y_test, preds, alpha=0.3, label=name)

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             color="black", linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_res
