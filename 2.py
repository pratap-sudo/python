import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Create a synthetic dataset for demonstration
print("1. Original Dataset:")
data = {
    'Feature1': [10, 20, None, 40, 50, 60, None, 80, 90, 100],
    'Feature2': [1.1, 2.2, 3.3, 4.4, None, 6.6, 7.7, 8.8, 9.9, 10.0],
    'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(df)
print("\n")

# Identify numerical and categorical features
numerical_features = ['Feature1', 'Feature2']
categorical_features = ['Category']

# 2. Define Preprocessing Steps for Numerical Features
#    - Imputation: Fill missing values with the mean
#    - Scaling: Standardize features (mean=0, variance=1)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Define Preprocessing Steps for Categorical Features
#    - One-Hot Encoding: Convert categorical variables into a numerical format
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Create a ColumnTransformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Apply the preprocessing steps
# Fit and transform the data
X_processed = preprocessor.fit_transform(df[numerical_features + categorical_features])

# Get feature names after one-hot encoding for categorical features
onehot_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
processed_feature_names = numerical_features + list(onehot_feature_names)

# Convert the processed array back to a DataFrame for better readability
df_processed = pd.DataFrame(X_processed, columns=processed_feature_names)

print("2. Processed Dataset:")
print(df_processed)
print("\n")

print("Data Preprocessing Summary:")
print("- Missing values in numerical features were imputed with the mean.")
print("- Numerical features were scaled using StandardScaler.")
print("- Categorical features were converted using One-Hot Encoding.")
