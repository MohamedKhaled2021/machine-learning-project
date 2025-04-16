# Ensure required libraries are installed
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    import numpy as np
except ImportError as e:
    print("Missing library:", e.name)
    print("Please install it using: pip install", e.name)
    exit(1)

# Improved dataset for better results
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Experience']]
y = df['Salary']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Cross-Validation
cv_scores = cross_val_score(model, X_poly, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Feature Importance Analysis
feature_importance = np.abs(model.coef_)
print("Feature Importance:", feature_importance)