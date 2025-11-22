import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pre_processing.clean_airbnb import clean_airbnb_raw
from scipy.stats import pearsonr

def generate_data_audit_report():
    print("Starting Data Audit...")

    # load raw data
    df_raw = pd.read_csv(file_path)
    n_initial = len(df_raw)
    print(f"Initial Observations: {n_initial}")

    # track data loss
    loss_log = [("Raw Data", n_initial)]

    df_temp = df_raw.drop_duplicates()
    loss_log.append(("Drop Duplicates", len(df_temp)))

    if "room type" in df_temp.columns:
        df_temp = df_temp[df_temp["room type"].isin(["Private room", "Entire home/apt"])]
    loss_log.append(("Filter Room Type", len(df_temp)))

    # run full cleaning pipeline
    print("\nRunning full cleaning pipeline (this may take a moment)...")
    df_final = clean_airbnb_raw(df_raw)

    print(f"\nFinal Cleaned Observations: {len(df_final)}")
    print(f"Total Data Retained: {len(df_final) / n_initial:.1%}")

    # ---- visualizations ----
    # FIG 1: Data Cleaning Funnel
    loss_df = pd.DataFrame(loss_log, columns=["Step", "Count"])
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=loss_df, x="Step", y="Count", palette="viridis")
    plt.title("Methodology: Data Reduction Pipeline", fontsize=16)
    plt.ylabel("Number of Observations")
    for i, v in enumerate(loss_df["Count"]):
        ax.text(i, v + 1000, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # FIG 2: Price Distribution (The Target)
    target_col = "price"
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df_final[target_col], bins=50, kde=True, color="teal")
    plt.title(f"Distribution of {target_col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_final[target_col], color="teal")
    plt.title(f"Boxplot of {target_col} (Outlier Check)")
    plt.tight_layout()
    plt.show()

    # FIG 3: Spatial Feature Analysis
    spatial_cols = [
        "dist_to_subway", "dist_to_midtown",
        "dist_to_park", "airbnb_density_250m"
    ]
    spatial_cols = [c for c in spatial_cols if c in df_final.columns]

    if spatial_cols:
        plt.figure(figsize=(10, 8))
        numeric_df = df_final.select_dtypes(include=[np.number])
        cols_to_corr = [target_col] + spatial_cols
        corr_mat = numeric_df[cols_to_corr].corr()

        sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix: Price vs. Spatial Features", fontsize=16)
        plt.tight_layout()
        plt.show()

    # FIG 4: service fee
    print("\nChecking Service Fee Redundancy...")
    check_df = df_raw.copy()
    for c in ["price", "service fee"]:
        if c in check_df.columns:
            check_df[c] = (
                check_df[c].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            check_df[c] = pd.to_numeric(check_df[c], errors="coerce")

    check_df = check_df.dropna(subset=["price", "service fee"])

    if not check_df.empty:
        corr, _ = pearsonr(check_df["price"], check_df["service fee"])
        print(f"Correlation (Price vs Service Fee): {corr:.4f}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=check_df["price"], y=check_df["service fee"], alpha=0.5, color='purple')
        plt.title(f"Redundancy Check: Service Fee vs Price\nCorrelation: {corr:.4f}", fontsize=14)
        plt.xlabel("Price ($)")
        plt.ylabel("Service Fee ($)")

        # add 20% line
        x_vals = np.array([0, check_df["price"].max()])
        y_vals = 0.2 * x_vals
        plt.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=2, label='Exact 20% Ratio')
        plt.legend()

        plt.tight_layout()
        plt.show()

file_path = r"/Users/anianvonmengershausen/PycharmProjects/airbnb/Airbnb_Open_Data.csv"
sns.set_style("whitegrid")
generate_data_audit_report()

