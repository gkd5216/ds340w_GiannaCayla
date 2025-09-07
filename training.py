import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_parquet("Datasets/no_cheater_present/0.parquet")
features = ["shots_fired", "armor_value", "velocity", "tick"]
label = "is_alive"

available_features = [f for f in features if f in df.columns]
if label not in df.columns:
    raise ValueError(f"Label column '{label}' not found in dataset")

df = df[available_features + [label]].dropna().head(2000)
X = df[available_features]
y = df[label].astype(int)

# Train/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(
    X_temp, y_temp, test_size=0.33, random_state=42)

print(X_train)
print(X_test)
print(X_validation)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", round(acc, 3))

y_pred = model.predict(X_validation)
acc = accuracy_score(y_validation, y_pred)
print("Validation Accuracy:", round(acc, 3))