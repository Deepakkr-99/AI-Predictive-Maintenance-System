import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/encoded_data.csv")

# Input and Output
x = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest Model
model = RandomForestClassifier(random_state=42)

# Parameters
param = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10]
}

# Hyperparameter Tuning
search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param,
    n_iter=5,
    cv=5,
    random_state=42
)

# Train Model
search.fit(x_train, y_train)

# Best Parameters
print("Best Parameters")
print(search.best_params_)

# Best Cross Validation Score
print("Best Cross Validation Score")
print(search.best_score_)

# Best Model
best_model = search.best_estimator_

# Prediction
pred = best_model.predict(x_test)

# Accuracy
acc = accuracy_score(y_test, pred)
print("Tuned Random Forest Accuracy")
print(acc)
