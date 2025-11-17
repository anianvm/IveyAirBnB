import pandas as pd
from sklearn.model_selection import train_test_split

from Anian.pre_processing.preprocess_for_regression import preprocess_for_regression
import Anian.supervised_models.regression_fast as rfast

# VISUALS
import Anian.supervised_models.regression_visuals as rvis               # :contentReference[oaicite:1]{index=1}


def run_fast_regression(path_to_csv):

    # 1. Load Data
    df_raw = pd.read_csv(path_to_csv)

    # 2. Preprocess (impute, winsor, scale, dummy)
    X_processed, y = preprocess_for_regression(df_raw, target_variable="price")
    feature_names = X_processed.columns

    # 3. Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    # ============================================================
    # 4. FAST SINGLE MODELS
    # ============================================================
    print("\nTraining Fast Tree…")
    tree = rfast.train_tree_fast(x_train, y_train)

    print("\nTraining Fast Random Forest…")
    forest = rfast.train_forest_fast(x_train, y_train)

    print("\nTraining Fast Gradient Boosting…")
    gboost = rfast.train_gboost_fast(x_train, y_train)

    print("\nTraining Fast Lasso…")
    lasso = rfast.train_lasso_fast(x_train, y_train)

    print("\nTraining Fast Ridge…")
    ridge = rfast.train_ridge_fast(x_train, y_train)

    print("\nTraining Fast ElasticNet…")
    elastic = rfast.train_elastic_fast(x_train, y_train)

    print("\nTraining Fast Linear SVR…")
    lsvr = rfast.train_linear_svr_fast(x_train, y_train)

    single_models = {
        "Decision Tree": tree,
        "Random Forest": forest,
        "GradientBoost": gboost,
        "Lasso": lasso,
        "Ridge": ridge,
        "ElasticNet": elastic,
        "LinearSVR": lsvr,
    }

    # ============================================================
    # 5. VISUALS FOR SINGLE MODELS
    # ============================================================
    print("\n=== TREE FEATURE IMPORTANCE ===")
    rvis.explain_tree_model(tree, feature_names)

    print("\n=== LASSO COEFFICIENTS ===")
    rvis.explain_regularized_model(lasso, feature_names)

    print("\n=== PERMUTATION IMPORTANCE (Forest) ===")
    rvis.explain_feature_importance(
        forest, x_train, y_train, x_test, y_test
    )

    print("\n=== COMPARISON OF SINGLE MODELS ===")
    rvis.compare_regression_models(single_models, x_test, y_test)

    # ============================================================
    # 6. ENSEMBLES (FAST)
    # ============================================================
    from regression_ensemble_fast import (
        bag_fast, vote_fast, stack_fast
    )

    print("\nBuilding Bagging Model…")
    bag = bag_fast(single_models, x_train, y_train)

    print("\nBuilding Voting Model…")
    vote = vote_fast(single_models)
    vote.fit(x_train, y_train)

    print("\nBuilding Stacking Model…")
    stack = stack_fast(single_models)
    stack.fit(x_train, y_train)

    ensemble_models = {
        "Bagging": bag,
        "Voting": vote,
        "Stacking": stack
    }

    print("\n=== COMPARISON OF ENSEMBLE MODELS ===")
    rvis.compare_regression_models(ensemble_models, x_test, y_test)

    print("\nDONE ✓")


# ============================================================
# 7. To actually run it:
# ============================================================
if __name__ == "__main__":
    run_fast_regression(
        "/Users/anianvonmengershausen/PycharmProjects/FinalAirBnB/Airbnb_Open_Data.csv"
    )
