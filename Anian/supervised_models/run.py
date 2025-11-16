import pandas as pd
from sklearn.model_selection import train_test_split

from Anian.pre_processing.preprocess_for_regression import preprocess_for_regression
import regression_single_models as rsm
import regression_ensemble_models as rem

df_raw = pd.read_csv(r"/Users/anianvonmengershausen/PycharmProjects/FinalAirBnB/Airbnb_Open_Data.csv")

# 1. Preprocess
X_processed, y = preprocess_for_regression(df_raw, target_variable="price")

# 2. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42
)

tree_rmodel = rsm.train_tree_model(x_train,y_train)
svm_rmodel = rsm.train_svm_model(x_train,y_train)
lasso_rmodel = rsm.train_lasso_model(x_train,y_train)

tree_best_params = tree_rmodel.get_params()
svm_best_params = svm_rmodel.get_params()
lasso_best_params = lasso_rmodel .get_params()

single_models = {
    "Decision Tree": tree_rmodel,
    "SVM": svm_rmodel,
    "LogisticReg Lasso": lasso_rmodel,
}

rsm.explain_tree_model(tree_rmodel,x_train)
rsm.explain_feature_importance(svm_rmodel,x_train, y_train, x_test, y_test)
rsm.explain_regularized_model(lasso_rmodel,x_train)
rsm.compare_regression_models(single_models,x_test, y_test)

rmodels = {
    "Decision Tree": DecisionTreeRegressor(**tree_best_params),
    "SVM": SVR(**svm_best_params),
    "LogisticReg Lasso": Lasso(**lasso_best_params),
}

best_model_name_bg, best_bagging_rmodel = rem.bag_models(rmodels,x_train,y_train, x_test, y_test)
best_model_name_bst,best_boosting_rmodel = rem.bag_models(rmodels,x_train, y_train, x_test, y_test)
soft_voting_rmodel = rem.vote_models(rmodels)
soft_voting_rmodel.fit(x_train,y_train)
meta_learner = MLPRegressor(
    hidden_layer_sizes=(50,),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)

stacking_rmodel=rem.stack_models(rmodels, meta_learner)
stacking_rmodel.fit(x_train,y_train)
ensemble_rmodels={
    "Bagging": best_bagging_rmodel,
    "Boosting": best_boosting_rmodel,
    'Voting': soft_voting_rmodel,
    'Stacking': stacking_rmodel
}

rsm.compare_regression_models(ensemble_rmodels,x_test, y_test)