import pandas as pd
import joblib

# Load model
model = joblib.load('model/model.pkl')

# Example input sample
sample = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Predict
prediction = model.predict(sample)[0]
print(f"Prediction: {prediction}")


