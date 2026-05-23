import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Data load karo
df = pd.read_csv("patient_data.csv")

# Features aur Label
X = df[["systolic", "diastolic", "hb", 
        "weeks", "history", "bleeding"]]
y = df["risk_label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model banao
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Accuracy check karo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.1f}%")

# Model save karo
joblib.dump(model, "maternal_model.pkl")
print("✅ Model saved!")