import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Column names for iris dataset
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Load dataset
df = pd.read_csv('data/iris.data', header=None, names=columns)

# Features and target
X = df.drop('class', axis=1)
y = df['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model (using k=5 as default)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the model
joblib.dump(knn, 'model/model.pkl')

print("✅ KNN model trained and saved to model/model.pkl")
