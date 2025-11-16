import pandas as pd
from data_preprocessing_project import *

data = pd.read_csv('Airbnb_Open_Data.csv')
data = remove_feature_substr(data,'price','$')
data = convert_feature_type(data,'price','int')
data = remove_feature_substr(data,'service fee','$')
data = convert_feature_type(data,'price','int')