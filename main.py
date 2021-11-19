import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle


#here uploding the data
data = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\practoce\data.csv')
#columns need more for the house
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'waterfront', 'sqft_above']
house = data[features]
val = data.price

#splinting our data sets
train_house ,test_house ,train_val ,test_val =train_test_split(house,val ,test_size=0.2 ,random_state= 10)



#creating our model
model = DecisionTreeRegressor(max_leaf_nodes=58,random_state=10)      #after many train I got random_state =10 is good

#fitting the model
model.fit(train_house,train_val)
prediction = model.predict(test_house)

filename = r'C:\Users\Lenovo\PycharmProjects\practoce\decitiontree.sav'
pickle.dump(model, open(filename,'wb'))
#error in the model

print(model.score(train_house,train_val))