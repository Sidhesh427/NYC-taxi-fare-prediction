# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
training_set = pd.read_csv('train.csv', nrows = 5000000)
test_set = pd.read_csv('test.csv')

training_set.info()
training_set.describe()
test_set.describe()
print(training_set.isnull().sum())
training_set = training_set.dropna(how = 'any', axis = 'rows')

def clean_df(df):
    return df[(df.fare_amount > 0) & 
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45) &
            (df.passenger_count > 0) & (df.passenger_count < 10)]

training_set = clean_df(training_set)
test_set = clean_df(test_set)

#Convert to date-time format
training_set['pickup_datetime'] = pd.to_datetime(training_set['pickup_datetime'])
training_set['hour'] = training_set.pickup_datetime.dt.hour
training_set['day'] = training_set.pickup_datetime.dt.day
training_set['month'] = training_set.pickup_datetime.dt.month
training_set['weekday'] = training_set.pickup_datetime.dt.weekday
training_set.drop(columns = ['key','pickup_datetime'], axis = 1, inplace = True)

test_set['pickup_datetime'] = pd.to_datetime(test_set['pickup_datetime'])
test_set['hour'] = test_set.pickup_datetime.dt.hour
test_set['day'] = test_set.pickup_datetime.dt.day
test_set['month'] = test_set.pickup_datetime.dt.month
test_set['weekday'] = test_set.pickup_datetime.dt.weekday
test_set.drop(columns = ['key','pickup_datetime'], axis = 1, inplace = True)

#Adding column for distance travelled between 2 points
def dist_btwn_2_pts(df):
    R = 6371
    phi_1 = np.radians(df['pickup_latitude'])
    phi_2 = np.radians(df['dropoff_latitude'])
    phi_change = np.radians(df['pickup_latitude'] - df['dropoff_latitude'])
    lambda_change = np.radians(df['pickup_longitude'] - df['dropoff_longitude'])
    a = np.sin(phi_change/2)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(lambda_change/2)**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return  R * c

d = dist_btwn_2_pts(training_set)
training_set['distance'] = d

d = dist_btwn_2_pts(test_set)
test_set['distance'] = d

# Splitting the dataset into the Training set and Test set
X = training_set.iloc[:, 1:].values
y = training_set.iloc[:,:1].values
test_set1 = test_set.iloc[:,0:].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #6.74

# Fitting XGBoost to the Training set
from xgboost import XGBRegressor
regressor1 = XGBRegressor(n_jobs = -1)
regressor1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = regressor.predict(X_test)
y_pred_test = regressor.predict(test_set1)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
accuracies.mean()

from sklearn import metrics
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #4.371



