import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
weather = pd.read_csv('weatherHistory.csv')


# transforming from qualitative feature into quantitative
print(weather.PrecipType.value_counts())
print("After we transformed:")
weather['PrecipType']=pd.get_dummies(weather.PrecipType,drop_first=True)
print(weather.PrecipType.value_counts())

##Null values
nulls = pd.DataFrame(weather.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

##handling missing value
data = weather.select_dtypes(include=[np.number]).interpolate().dropna()
print("the amount of missing data are")
print(sum(data.isnull().sum() != 0))

##Build a linear model
X = weather[['Visibility','WindBearing','PrecipType']]
print(X)
y=weather['Temperature']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)  #create the training dataset and test dataset
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
# ##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
