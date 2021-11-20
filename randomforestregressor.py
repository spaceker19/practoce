import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle


#here uploding the data
data = pd.read_csv('data.csv')
#columns need more for the house
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','floors', 'sqft_above']
house = data[features]
val = data.price

#splinting our data sets
train_house ,test_house ,train_val ,test_val =train_test_split(house,val ,test_size=0.2 ,random_state= 10)

#creating our model
modal = RandomForestRegressor(n_estimators = 300,random_state=0)

#fitting the model
modal.fit(train_house,train_val)
prediction = modal.predict(test_house)


filename = 'randomforestregressor.sav'
pickle.dump(modal, open(filename, 'wb'))
error = mean_absolute_error(test_val,prediction)















# print(error)                                                    #181392.44145439603
# print(modal.score(train_house,train_val))                       #0.8666505935661103



# In this regressor it's take more time as compered to DecisionTree beacuse of this model contained maltipule
# DecisionTree.