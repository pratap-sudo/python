# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Step 2: Create dataset
data = {
    'Age': [20, 22, 25, 30, 35, 40, 45, 50],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    'Buy': [0, 0, 0, 1, 1, 1, 1, 1]   # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 3: Split features and target
X = df[['Age', 'Salary']]
y = df['Buy']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict
prediction = model.predict(X_test)

print("Predictions:", prediction)
print("Actual:", y_test.values)

# Step 8: Predict new data
new_data = [[28, 32000]]
result = model.predict(new_data)
print("Prediction for new data:", result[0])
