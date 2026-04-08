from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("3. Implementing Classification Algorithm (Logistic Regression):")

# Separate features (X) and target (y)
# X is our processed features (df_processed)
# y is the original target column from the unprocessed dataframe
X = df_processed
y = df['Target']

# Split the data into training and testing sets
# We'll use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42, solver='liblinear') # Using 'liblinear' solver for small datasets

# Train the model using the training data
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

print("\nClassification Algorithm Implementation Summary:")
print("- Data was split into training and testing sets (80/20 split).")
print("- A Logistic Regression model was initialized and trained on the training data.")
print("- The model's performance was evaluated on the test set using accuracy and a classification report.")
