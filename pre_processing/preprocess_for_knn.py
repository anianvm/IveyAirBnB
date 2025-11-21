from pre_processing.clean_airbnb import clean_airbnb_raw
from feature_engine.outliers import Winsorizer
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def preprocess_for_knn(df_raw):
    df = clean_airbnb_raw(df_raw)

    cat = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    num = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[cat] = df[cat].astype(str)

    # Normalize columns & add hot-encoder specifically for kmeans instead of dummy
    pipeline = Pipeline([
        ('encoder', OneHotEncoder(variables=cat, drop_last=True)),
        ('winsor', Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=num)),
        ('scaler', SklearnTransformerWrapper(transformer=StandardScaler(), variables=num)),
    ])

    # Fit and transform the df_proc DataFrame
    df_processed = pipeline.fit_transform(df)

    return df
