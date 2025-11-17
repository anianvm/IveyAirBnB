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

    # 1. Log-transform the target variable
    y = np.log1p(y)

    # 2. Detect types
    cat = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 3. Convert booleans to string
    X[cat] = X[cat].astype(str)

    # 4. Create dummy variables FIRST (from X) and store in X_proc
    X_proc = pd.get_dummies(X, columns=cat, drop_first=True)

    # 5. NOW create the pipeline to scale ONLY the numerical columns
    #    (Removing the redundant MeanMedianImputer)
    pipeline = Pipeline([
        # --- THIS IS THE FIX ---
        ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=num)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num)),
    ])

    # 6. Fit and transform the X_proc DataFrame
    X_proc = pipeline.fit_transform(X_proc)

    return X_proc, y