import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier


# Ensure model folder exists
os.makedirs("model", exist_ok=True)


# Load dataset
df = pd.read_csv("data/clean_churn.csv")


# Split features and target
X = df.drop("churn", axis=1)
y = df["churn"]


# Save feature names (IMPORTANT for API)
with open("model/columns.json", "w") as f:
    json.dump(list(X.columns), f)

print("Columns saved.")


# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Compute imbalance weight
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)


# Build model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Save trained model
joblib.dump(model, "model/churn_xgb.pkl")
print("Model saved.")


# Evaluation (for verification only)

y_prob = model.predict_proba(X_test)[:, 1]

t = 0.8
y_custom = (y_prob > t).astype(int)

print("\nThreshold:", t)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_custom))

print("\nClassification Report:")
print(classification_report(y_test, y_custom))

roc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC:", roc)
