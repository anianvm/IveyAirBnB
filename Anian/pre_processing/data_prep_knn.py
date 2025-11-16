import pandas as pd
from sklearn.preprocessing import StandardScaler
from clean_airbnb import clean_airbnb_raw

def preprocess_for_knn(df_raw):
    df = clean_airbnb_raw(df_raw)

    # --- Dummy encoding first ---
    columns_to_dummy = [
        'host_identity_verified',
        'neighbourhood group',
        'instant_bookable',
        'cancellation_policy',
        'room type'
    ]

    df[columns_to_dummy] = df[columns_to_dummy].astype(str)  # avoids boolean issues

    df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)

    # --- Now scale EVERYTHING ---
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df
