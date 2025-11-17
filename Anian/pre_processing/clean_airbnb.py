import numpy as np
import pandas as pd
from haversine import haversine


def clean_airbnb_raw(df):

    df = df.copy()

    # --- DEDUPE --- #
    df = df.drop_duplicates()

    # --- CLEAN PRICE + SERVICE FEE --- #
    for col in ["price", "service fee"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- FORCE NUMERIC TYPES --- #
    numeric_cols_to_force = [
        "minimum nights", "availability 365", "Construction year",
        "number of reviews", "reviews per month",
        "review rate number", "calculated host listings count"
    ]
    for col in numeric_cols_to_force:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- CLIP EXTREMES ---
    df["availability 365"] = df["availability 365"].clip(0, 365)
    df["minimum nights"] = df["minimum nights"].clip(0, 90)

    # --- LOG TRANSFORMS (reduce skew) ---
    df["minimum nights"] = np.log1p(df["minimum nights"])
    df["availability 365"] = np.log1p(df["availability 365"])
    df["number of reviews"] = np.log1p(df["number of reviews"])
    df["calculated host listings count"] = np.log1p(df["calculated host listings count"])

    # --- HAVERSINE DISTANCE --- #
    ts_lat, ts_long = 40.7580, -73.9855
    if "lat" in df.columns and "long" in df.columns:
        df["dist_to_times_sq"] = df.apply(
            lambda r: haversine((r["lat"], r["long"]), (ts_lat, ts_long)),
            axis=1
        )

    # --- DROP UNUSED COLUMNS --- #
    cols_to_drop = [
        "host id", "id", "NAME", "name", "host name", "neighbourhood",
        "lat", "long", "country", "country code", "house_rules",
        "license", "last review", "reviews per month", "review rate number", "service fee"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # --- FIX NEIGHBOURHOOD GROUP --- #
    df["neighbourhood group"] = (
        df["neighbourhood group"]
        .replace({"Brookln": "Brooklyn", "Manhatan": "Manhattan", "brookln": "Brooklyn"})
        .fillna("Missing")
    )

    # Remove boroughs we don't want
    df = df[df["neighbourhood group"] != "Staten Island"] # Bronx

    # --- FILTER ROOM TYPES --- #
    df = df[df["room type"].isin(["Private room", "Entire home/apt"])]

    # --- HANDLE MISSING VALUES --- #
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("Missing")

    # Remove rows that still have Missing values
    # df = df[~df.eq("Missing").any(axis=1)]

    return df
