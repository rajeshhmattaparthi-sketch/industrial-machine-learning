"""
Industrial Machine Learning - Predictive Maintenance (Simple Base Project)

Goal: Predict whether an industrial machine will FAIL, using sensor readings
(temperature, vibration, pressure, rotational speed, operating hours).

Run:  pip install pandas numpy scikit-learn matplotlib joblib
      python industrial_ml.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------
# 1. Create (simulate) industrial sensor data
#    Replace this with: df = pd.read_csv("your_machine_data.csv")
# ---------------------------------------------------------------
def generate_sensor_data(n=5000):
    df = pd.DataFrame({
        "temperature":     rng.normal(75, 10, n),      # deg C
        "vibration":       rng.normal(0.5, 0.15, n),   # mm/s
        "pressure":        rng.normal(30, 5, n),       # bar
        "rotation_speed":  rng.normal(1500, 150, n),   # RPM
        "operating_hours": rng.uniform(0, 10000, n),   # hours
    })

    # Failure risk rises with heat, vibration, pressure and wear
    risk = (
        0.05 * (df["temperature"] - 75)
        + 4.0 * (df["vibration"] - 0.5)
        + 0.04 * (df["pressure"] - 30)
        + 0.0003 * (df["operating_hours"] - 5000)
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-(risk * 2 - 2.5)))
    df["failure"] = (rng.random(n) < prob).astype(int)
    return df


df = generate_sensor_data()
print("Dataset shape:", df.shape)
print("Failure rate: {:.1%}".format(df["failure"].mean()))
print(df.head(), "\n")

# ---------------------------------------------------------------
# 2. Split features / target and train-test split
# ---------------------------------------------------------------
X = df.drop(columns="failure")
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------
# 3. Build and train the model
# ---------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
    )),
])
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC: {:.3f}\n".format(roc_auc_score(y_test, y_prob)))

# ---------------------------------------------------------------
# 5. Feature importance plot
# ---------------------------------------------------------------
importances = model.named_steps["clf"].feature_importances_
order = np.argsort(importances)
plt.figure(figsize=(7, 4))
plt.barh(X.columns[order], importances[order], color="steelblue")
plt.title("Which sensors predict machine failure?")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
print("Saved plot: feature_importance.png")

# ---------------------------------------------------------------
# 6. Save the model and predict on a new machine reading
# ---------------------------------------------------------------
joblib.dump(model, "maintenance_model.joblib")
print("Saved model: maintenance_model.joblib")

new_reading = pd.DataFrame([{
    "temperature": 92, "vibration": 0.85, "pressure": 36,
    "rotation_speed": 1450, "operating_hours": 8800,
}])
p = model.predict_proba(new_reading)[0, 1]
print("\nNew machine reading -> failure probability: {:.1%}".format(p))
print("Action:", "SCHEDULE MAINTENANCE" if p > 0.5 else "Machine OK")
