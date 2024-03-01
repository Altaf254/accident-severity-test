
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


np.random.seed(0)
n_samples = 1000
road_conditions = np.random.randint(1, 4, n_samples)  # 1 for good, 2 for fair, 3 for poor
weather = np.random.randint(1, 4, n_samples)  # 1 for clear, 2 for rainy, 3 for snowy
vehicle_type = np.random.randint(1, 4, n_samples)  # 1 for car, 2 for truck, 3 for motorcycle
speed_limit = np.random.randint(30, 300, n_samples)  # in kph
severity = np.random.randint(1, 6, n_samples)  # 1 to 5 representing severity levels


data = pd.DataFrame({'Road_Conditions': road_conditions,
                     'Weather': weather,
                     'Vehicle_Type': vehicle_type,
                     'Speed_Limit': speed_limit,
                     'Severity': severity})


X = data[['Road_Conditions', 'Weather', 'Vehicle_Type', 'Speed_Limit']]
y = data['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'accident_severity_model.pkl')
