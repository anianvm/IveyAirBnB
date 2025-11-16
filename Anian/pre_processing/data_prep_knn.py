import pandas as pd
from sklearn.preprocessing import StandardScaler
from clean_airbnb import clean_airbnb_raw

def preprocess_for_knn(df_raw):
    df = clean_airbnb_raw(df_raw)

    # dummy encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # scale everything (required for KNN + KMeans)
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df
