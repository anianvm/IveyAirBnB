import pandas as pd
from Anian.pre_processing.data_prep_knn import data_prep
from supervised_models import prepare_regression_data, train_all_models

df_raw = pd.read_csv(
    "/Users/anianvonmengershausen/PycharmProjects/FinalAirBnB/Airbnb_Open_Data.csv",
    low_memory=False,
)


df_model = data_prep(df_raw, categories=False)
print(df_model.head())
"""
df_model, labels = kmeans_analysis(df_model, k=6, make_plots=False)

export_kmeans_report(
    df_with_clusters = df_model,
    k=6,
    save_path="/Users/anianvonmengershausen/Desktop/CBS/Ivey/03_BusinessProgramming/Airbnb"
)
"""

X, y = prepare_regression_data(
    df_model,
    target_variable="price"   # flexible target variable
)
results, importances = train_all_models(X, y)

print("\nMODEL PERFORMANCE:")
print(results)
