import pandas as pd
import numpy as np
import joblib
import urllib.request
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── Download CSV if not already present ──────────────────────
import os
if not os.path.exists("titanic.csv"):
    print("Downloading titanic.csv...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "titanic.csv"
    )
    print("Downloaded.")

# ── Load & clean ──────────────────────────────────────────────
df = pd.read_csv("titanic.csv")
df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].dropna()

df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
y = df["Survived"].values

# ── Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ── Evaluate ──────────────────────────────────────────────────
acc = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.3f}")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(model,  "titanic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Done — titanic_model.pkl and scaler.pkl saved.")
