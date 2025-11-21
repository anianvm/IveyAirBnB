import pandas as pd
import numpy as np
import osmnx as ox
from sklearn.neighbors import NearestNeighbors

def fetch_osm_data(place, tags):
    # OSMnx v2.0+ uses features_from_place
    if hasattr(ox, 'features_from_place'):
        return ox.features_from_place(place, tags=tags)
    # OSMnx v1.x uses geometries_from_place
    elif hasattr(ox, 'geometries_from_place'):
        return ox.geometries_from_place(place, tags=tags)
    else:
        raise ImportError("Your version of osmnx is too old or incompatible.")

def robust_osm_fetch(tags, place_name="New York City, USA"):
    print(f"Fetching OSM data for {tags}...")

    boroughs = [
        "Manhattan, New York, NY, USA",
        "Brooklyn, New York, NY, USA",
        "Queens, New York, NY, USA",
        "The Bronx, New York, NY, USA",
        "Staten Island, New York, NY, USA"
    ]

    gdf = pd.DataFrame()

    # Change to go directly to borough method
    try:
        gdf = fetch_osm_data(place_name, tags)
    except Exception as e:
        print(f"  > Direct query for {place_name} failed. Retrying by borough...")
        gdfs = []
        for borough in boroughs:
            try:
                sub_gdf = fetch_osm_data(borough, tags)
                if not sub_gdf.empty:
                    gdfs.append(sub_gdf)
            except Exception as err:
                # Just log warning, don't crash if one borough fails
                print(f"  > Warning: Could not fetch {borough} ({err})")

        if not gdfs:
            print(f"  > FAILED: All queries failed for tags={tags}")
            return pd.DataFrame()  # Return empty so pipeline continues

        gdf = pd.concat(gdfs)

    # Find centroids for polygon structures (applies to big parks etc.)
    gdf = gdf.to_crs(epsg=4326)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["long"] = gdf.geometry.centroid.x

    # Drop any that failed conversion
    return gdf[["lat", "long"]].dropna()

def strict_nearest_distance(df, poi_df, new_col):
    df = df.copy()
    if poi_df.empty:
        print(f"  > Warning: No data found for {new_col}, filling with NaN")
        df[new_col] = np.nan
        return df

    # Convert to Radians for Haversine
    # Filter out any NaN (just to be safe...)
    mask = df['lat'].notna() & df['long'].notna()

    if not mask.any():
        df[new_col] = np.nan
        return df

    df_rad = np.radians(df.loc[mask, ["lat", "long"]].values)
    poi_rad = np.radians(poi_df[["lat", "long"]].values)

    # Fit Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=1, metric='haversine', algorithm='ball_tree')
    nbrs.fit(poi_rad)

    distances, _ = nbrs.kneighbors(df_rad)

    # multiply by earth radius
    df.loc[mask, new_col] = distances[:, 0] * 6371000
    return df

def compute_airbnb_density(df):
    df = df.copy()

    # only get valid lat and long data
    mask = df['lat'].notna() & df['long'].notna()
    coords_rad = np.radians(df.loc[mask, ['lat', 'long']].values)

    # distance to next airbnb
    nbrs = NearestNeighbors(n_neighbors=2, metric='haversine', algorithm='ball_tree')
    nbrs.fit(coords_rad)
    distances, _ = nbrs.kneighbors(coords_rad)

    df.loc[mask, 'dist_to_nearest_airbnb'] = distances[:, 1] * 6371000

    # Radius 250m
    radius_meters = 250
    radius_rad = radius_meters / 6371000
    count_neighbors = nbrs.radius_neighbors(coords_rad, radius=radius_rad, return_distance=False)

    df.loc[mask, 'airbnb_density_250m'] = [len(x) - 1 for x in count_neighbors]

    return df

# run function
def compute_open_data_features(df):
    df = df.copy()

    # Distance to other airbnbs
    print("Computing Airbnb density features...")
    try:
        df = compute_airbnb_density(df)
    except Exception as e:
        print(f"Density computation failed: {e}")

    ## Manual distance to mid-town
    # Distance to Empire State Building
    print("Computing distance to Midtown...")
    midtown_df = pd.DataFrame({'lat': [40.748817], 'long': [-73.985428]})
    df = strict_nearest_distance(df, midtown_df, "dist_to_midtown")

    ## Osm features
    # Subways
    subways = robust_osm_fetch({"railway": "subway_entrance"})
    df = strict_nearest_distance(df, subways, "dist_to_subway")

    # Parks
    parks = robust_osm_fetch({"leisure": "park"})
    df = strict_nearest_distance(df, parks, "dist_to_park")

    # Restaurants
    restaurants = robust_osm_fetch({"amenity": "restaurant"})
    df = strict_nearest_distance(df, restaurants, "dist_to_rest")

    # Museums
    museums = robust_osm_fetch({"tourism": "museum"})
    df = strict_nearest_distance(df, museums, "dist_to_museum")

    # General Attractions
    attractions = robust_osm_fetch({"tourism": "attraction"})
    df = strict_nearest_distance(df, attractions, "dist_to_attraction")

    return df