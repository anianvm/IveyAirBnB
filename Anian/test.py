import matplotlib.pyplot as plt
import seaborn as sns
from Anian.pre_processing.data_prep_knn import *


def plot_clean_data_distributions(df):
    """
    Plot distribution histograms of key variables in the cleaned dataset.
    """

    vars_to_plot = {
        "dist_to_times_sq": "Distance to Times Square",
        "number of reviews": "Number of Reviews",
        "minimum nights": "Minimum Nights",
        "calculated host listings count": "Calculated Host Listings Count"
    }

    for col, title in vars_to_plot.items():
        if col not in df.columns:
            print(f"Skipping {col}: not in dataframe.")
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"Distribution of {title}")
        plt.xlabel(title)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


# load data
df_raw = pd.read_csv("/Users/anianvonmengershausen/PycharmProjects/FinalAirBnB/Airbnb_Open_Data.csv", low_memory=False)

df_raw = df_raw.drop_duplicates()
print(df_raw["neighbourhood group"].unique())
print(df_raw["neighbourhood group"].value_counts())


# clean data
df_clean = data_prep(df_raw)

# plot key distributions BEFORE kmeans
plot_clean_data_distributions(df_clean)

df_raw = df_raw.drop_duplicates()
print(df_raw["neighbourhood group"].unique())