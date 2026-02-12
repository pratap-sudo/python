# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Create a sample dataset
data = {
    'Age': [25, 30, np.nan, 45, 35],
    'Salary': [50000, 60000, 55000, np.nan, 65000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# -------------------------------
# 1. Handling Missing Values
# -------------------------------
imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

# -------------------------------
# 2. Encoding Categorical Data
# -------------------------------
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Purchased'] = label_encoder.fit_transform(df['Purchased'])

# -------------------------------
# 3. Feature Scaling
# -------------------------------
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# -------------------------------
# 4. Feature Selection (X and y)
# -------------------------------
X = df.drop('Purchased', axis=1)
y = df['Purchased']---

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nProcessed Dataset:\n", df)
print("\nTraining Features:\n", X_train)
print("\nTesting Features:\n", X_test)

