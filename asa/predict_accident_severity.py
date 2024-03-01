
import numpy as np
import joblib


loaded_model = joblib.load('accident_severity_model.pkl')

# Example prediction
example_data = np.array([[3, 2, 1, 210]])  # Assuming poor road conditions, rainny weather, car type, and 210 kph 
predicted_severity = loaded_model.predict(example_data)
print("Predicted severity:", predicted_severity[0])
