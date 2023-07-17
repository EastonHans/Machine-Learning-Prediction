import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("MET.csv")

# Select specific columns for X
X = data[['Humidity', 'Reading', 'Temp.', 'Wind dir.']]
y = data['Wind speed']

# Handle missing values in the feature matrix
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)

pickle.dump(rf_reg, open('model.pkl', 'wb'))

inputt = [float(x) for x in "15 1014 9.5 311".split(' ')]

final = [np.array(inputt)]

b = rf_reg.predict(final)

model = pickle.load(open('model.pkl', 'rb'))
