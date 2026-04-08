# Step 1: Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Step 2: Create dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5],
    'SleepHours': [5, 6, 7, 8, 9],
    'Attendance': [60, 70, 80, 90, 100],
    'Result': [0, 0, 1, 1, 1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Step 3: Split features and target
X = df[['StudyHours', 'SleepHours', 'Attendance']]
y = df['Result']

# Step 4: Train model
model = RandomForestClassifier()
model.fit(X, y)

# Step 5: Get feature importance
importance = model.feature_importances_

# Step 6: Display importance
for i, col in enumerate(X.columns):
    print(f"{col}: {importance[i]}")

# Step 7: Plot graph
plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
