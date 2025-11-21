from Anian.pre_processing.clean_airbnb import clean_airbnb_raw
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

def preprocess_for_regression(df_raw, target_variable):
    df = clean_airbnb_raw(df_raw)

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # detect data types
    cat = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # convert booleans to str to be detectable as a category
    X[cat] = X[cat].astype(str)

    # Create dummy columns
    X_proc = pd.get_dummies(X, columns=cat, drop_first=True)

    # Normalize numerical columns
    pipeline = Pipeline([
        ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=num)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num)),
    ])

    # Fit and transform the X_proc DataFrame
    X_proc = pipeline.fit_transform(X_proc)

    return X_proc, y