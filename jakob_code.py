import pandas as pd
import numpy
from sklearn.cluster import KMeans
from sklearn.preprocessing import *
from data_preprocessing_project import *

def data_prep(data):
    data = remove_feature_substr(data,'price','$')
    data = convert_feature_type(data,'price','int')
    data = remove_feature_substr(data,'service fee','$')
    data = convert_feature_type(data,'price','int')

    # Removing unnecessary columns
    column_names_index = data.columns
    print(column_names_index)
    columns_to_drop = ['host id','id','NAME','host name','neighbourhood','lat','long','country','country code','house_rules','license','last review','reviews per month']
    df = pd.DataFrame(data)
    df = df.drop(columns_to_drop, axis=1)

    # Finding number of NaN values in each column
    column_names_index = df.columns
    nan_count_col1 = df[column_names_index].isnull().sum()
    print(f"Number of NaNs: {nan_count_col1}")

    # Dropping whole row when NaN in one of the columns
    num_rows_og = len(df)
    print(f'Dataset contained {num_rows_og} rows')
    df = df.dropna()
    num_rows_new = len(df)
    print(f'Dropping rows with NaN values, {num_rows_og-num_rows_new} removed')

    # Remove commas
    df['price'] = df['price'].str.replace(',','').astype(int)
    df['service fee'] = df['service fee'].str.replace(',','').astype(int)


    # Checking for duplicate rows and removing them
    duplicate_rows = df.duplicated()
    print(f"Number of duplicate rows: {duplicate_rows.sum()}")
    df.drop_duplicates(inplace=True)
    print(f'Removing duplicates')

    # Printing columns and data types
    print(df.dtypes)

    # Make categorical variables into dummy variables
    columns_to_dummy = [
        'host_identity_verified',
        'neighbourhood group',
        'instant_bookable',
        'cancellation_policy',
        'room type'
        ]
    df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)

    # Normalize the values for each numerical variable
    scaler = StandardScaler()
    columns_to_normalize = [
        'price',
        'service fee',
        'review rate number',
        'Construction year',
        'minimum nights',
        'number of reviews',
        'calculated host listings count',
        'availability 365'
        ]
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    column_names_index = df.columns
    print(column_names_index)
    return df

def kmeans(df):

    selected_data = clean_df.iloc[:, 1:]

    # Determine how many clusters would be optimal - Elbow Method
    # Within-Cluster Sum of Squares WCSS
    wcss = []
    for i in range(1, 16):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(selected_data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 16), wcss)
    plt.title("The Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS-inertia")
    plt.show()

    k1 = 6
    kmeans_1 = KMeans(n_clusters=k1, init='k-means++', random_state=0)
    kmeans_1.fit_predict(selected_data)
    selected_data["Cluster"] = kmeans_1.labels_

    selected_data["Cluster"] = kmeans_1.labels_
    cluster_counts = selected_data['Cluster'].value_counts()
    print("\nNumber of points in each cluster:")
    print(cluster_counts)

    plt.figure(figsize=(16, 10))
    correlation = selected_data.corr()
    correlation['Cluster'].sort_values().plot(kind='bar')
    plt.xticks(rotation=45)
    plt.show()

    price = selected_data.groupby("Cluster").describe()
    sns.boxplot(x="Cluster",y="price", data=selected_data)
    plt.show()

    review_rate = selected_data.groupby("Cluster").describe()
    sns.boxplot(x="Cluster",y="review rate number", data=selected_data)
    plt.show()

    availability = selected_data.groupby("Cluster").describe()
    sns.boxplot(x="Cluster",y="availability 365", data=selected_data)
    plt.show()

    return df

data = pd.read_csv(r'C:\Users\jakob\Downloads\Airbnb_Open_Data.csv')
clean_df = data_prep(data)
kmeans(clean_df)