import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from Anian.pre_processing.preprocess_for_regression import preprocess_for_regression
from Anian.pre_processing.clean_airbnb import clean_airbnb_raw
import Anian.supervised_models.regression_fast as rfast
import Anian.supervised_models.regression_visuals as rvis
from Anian.supervised_models.regression_fast import (
    bag_fast, vote_fast, stack_fast
)


def analyze_spatial_features(df):
    # 1. Identify the new columns we care about
    # (Adjust names if yours are slightly different)
    new_cols = [
        'dist_to_nearest_airbnb',
        'airbnb_density_250m',
        'dist_to_subway',
        'dist_to_park',
        'dist_to_rest',
        'dist_to_museum',
        'dist_to_attraction',
        'dist_to_midtown'
    ]

    # Filter to only columns that actually exist in the dataframe
    present_cols = [c for c in new_cols if c in df.columns]

    if not present_cols:
        print("No spatial feature columns found!")
        return

    # 2. Print Descriptive Statistics
    print("-" * 40)
    print("Descriptive Statistics for Spatial Features")
    print("-" * 40)
    # 'describe' gives count, mean, std, min, 25%, 50%, 75%, max
    stats = df[present_cols].describe().round(2)
    print(stats)
    print("-" * 40)

    # 3. Plot Histograms
    # Set style
    sns.set_style("whitegrid")

    # Determine layout (rows x cols)
    num_plots = len(present_cols)
    cols = 2
    rows = (num_plots + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(present_cols):
        ax = axes[i]

        # Plot histogram with KDE (Kernel Density Estimate)
        # We drop NaNs specifically for plotting to avoid errors
        data_to_plot = df[col].dropna()

        sns.histplot(data_to_plot, kde=True, ax=ax, color='teal', bins=40)

        # Formatting
        ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value (Meters or Count)')
        ax.set_ylabel('Frequency')

        # Add a vertical line for the median
        median_val = data_to_plot.median()
        ax.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.1f}')
        ax.legend()

    # Hide any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# --- RUN IT ---

df_raw = pd.read_csv(
    r"/Users/anianvonmengershausen/PycharmProjects/FinalAirBnB/Airbnb_Open_Data.csv"
)
print(df_raw.describe())
df_cleaned = clean_airbnb_raw(df_raw)
print(df_cleaned.describe())
analyze_spatial_features(df_cleaned)

