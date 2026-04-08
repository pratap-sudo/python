import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# STEP 1 — Create Dataset
# ===============================

data = {
    'student': ['A','B','C'],
    'math': [2,0,4],
    'science': [0,1,3]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

plt.figure(figsize=(6,6))
plt.scatter(df['math'], df['science'])

for i in range(len(df)):
    plt.text(df['math'][i]+0.05,
             df['science'][i]+0.05,
             df['student'][i])

plt.xlabel("Math")
plt.ylabel("Science")
plt.title("Step 1: Original Data")
plt.grid(True)
plt.show()


# ===============================
# STEP 2 — Mean Centering
# ===============================

mean_math = df['math'].mean()
mean_science = df['science'].mean()

print("\nMean Math =", mean_math)
print("Mean Science =", mean_science)

df_centered = df.copy()
df_centered['math'] -= mean_math
df_centered['science'] -= mean_science

print("\nCentered Data:\n")
print(df_centered)

plt.figure(figsize=(6,6))
plt.scatter(df_centered['math'], df_centered['science'])

for i in range(len(df_centered)):
    plt.text(df_centered['math'][i]+0.05,
             df_centered['science'][i]+0.05,
             df_centered['student'][i])

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.xlabel("Math (Centered)")
plt.ylabel("Science (Centered)")
plt.title("Step 2: Mean Centered Data")
plt.grid(True)
plt.show()


# ===============================
# STEP 3 — Covariance Matrix
# ===============================

numeric_data = df_centered[['math','science']]
cov_matrix = np.cov(numeric_data.T)

print("\nCovariance Matrix:\n")
print(cov_matrix)


# ===============================
# STEP 4 — Eigenvalues & Eigenvectors
# ===============================

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)


# ===============================
# STEP 5 — Draw Principal Component
# ===============================

max_index = np.argmax(eigenvalues)
principal_component = eigenvectors[:, max_index]

print("\nPrincipal Component (PC1):\n", principal_component)

plt.figure(figsize=(6,6))
plt.scatter(df_centered['math'], df_centered['science'])

for i in range(len(df_centered)):
    plt.text(df_centered['math'][i]+0.05,
             df_centered['science'][i]+0.05,
             df_centered['student'][i])

origin = [0, 0]
plt.quiver(*origin,
           principal_component[0],
           principal_component[1],
           scale=3)

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("Step 5: Principal Component Direction")
plt.grid(True)
plt.show()


# ===============================
# STEP 6 — Project Data onto PC1
# ===============================

pca_scores = numeric_data.dot(principal_component)

print("\nPCA 1D Projection:\n")
print(pca_scores)


# ===============================
# STEP 7 — Visualize 2D → 1D Collapse
# ===============================

plt.figure(figsize=(8,2))
plt.scatter(pca_scores, [0]*len(pca_scores))

for i in range(len(pca_scores)):
    plt.text(pca_scores[i], 0.02, df['student'][i])

plt.title("Final Result: 2D → 1D PCA Projection")
plt.yticks([])
plt.grid(True)
plt.show()


# ===============================
# STEP 8 — Explained Variance Ratio
# ===============================

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

print("\nExplained Variance Ratio:\n", explained_variance_ratio)
