from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load data
iris = load_iris()

X = iris.data
y = iris.target


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)


# Evaluate
accuracy = model.score(X_test, y_test)

print(f"Model accuracy: {accuracy:.2%}")


# Save model
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(model, models_dir / "iris_model.joblib")

print("Model saved to models/iris_model.joblib")