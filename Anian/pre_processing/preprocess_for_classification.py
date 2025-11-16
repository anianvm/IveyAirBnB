from clean_airbnb import clean_airbnb_raw
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def preprocess_for_classification(df_raw, target_class):
    df = clean_airbnb_raw(df_raw)

    X = df.drop(columns=[target_class])
    y = df[target_class]

    cat = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # convert booleans to string
    X[cat] = X[cat].astype(str)

    pipeline = Pipeline([
        ('imputer', MeanMedianImputer(imputation_method='median', variables=num)),
        ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold='auto', variables=num)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num)),
    ])

    X_proc = pipeline.fit_transform(X)

    # ---> Add dummy encoding
    X_proc = pd.get_dummies(X_proc, columns=cat, drop_first=True)

    return X_proc, y
