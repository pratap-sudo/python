# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create dataset (Experience vs Salary)
data = {
    'Experience': [1, 2, 3, 4, 5],
    'Salary': [20000, 25000, 30000, 35000, 40000]
}

df = pd.DataFrame(data)

# Step 3: Define input (X) and output (y)
X = df[['Experience']]
y = df['Salary']

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predict value
predicted_salary = model.predict([[6]])
print("Predicted Salary for 6 years experience:", predicted_salary[0])

# Step 7: Plot graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Example")
plt.show()
