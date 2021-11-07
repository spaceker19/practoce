import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('data.csv')
# print(data.columns)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront', 'sqft_above']
house = data[features]
val = data.price



model = DecisionTreeRegressor(random_state=1)
model.fit(house,val)
result = model.predict(house)
print(result)