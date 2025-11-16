from clean_airbnb import clean_airbnb_raw
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def preprocess_for_classification(df_raw, target_class):
    df = clean_airbnb_raw(df_raw)

    X = df.drop(columns=[target_class])
    y = df[target_class]

    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    pipeline = Pipeline([
        ('imputer', MeanMedianImputer(imputation_method='median', variables=num)),
        ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold='auto', variables=num)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num)),
        # ('encoder', OneHotEncoder(variables=cat, drop_last=True))
    ])

    X_proc = pipeline.fit_transform(X)
    return X_proc, y
