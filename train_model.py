"""
train_model.py
Loads the Iris dataset, preprocesses it, trains a classification model,
evaluates it, and saves the trained model using Joblib.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("Dataset shape:", X.shape)
print(X.head())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Save model using Joblib
joblib.dump(model, "iris_model.pkl")
print("\nModel saved as iris_model.pkl")

# Save target names too, so the API can return readable labels
joblib.dump(list(iris.target_names), "target_names.pkl")
print("Target names saved as target_names.pkl")
