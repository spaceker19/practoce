import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import pickle
data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\practoce\data.csv')

object_data = data.dtypes == object
object_data = list(object_data[object_data].index)
data = data.drop(object_data,axis = 1)

val = data.price
data = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors',  'sqft_above']]

train_house ,test_house ,train_val ,test_val = train_test_split(data,val,train_size=0.8,random_state=0)

print(train_house.shape)
print(test_val.shape)

modal = XGBRegressor(n_estimators=1000)  #0.9999922692299098
modal.fit(train_house,train_val )


modal.save_model(r"C:\Users\Lenovo\PycharmProjects\practoce\xgboost.model")

print(modal.score(train_house,train_val))
