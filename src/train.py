# Load necessary libraries
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

DATASET_ROOT = os.path.expanduser("~/CS2CD.Counter-Strike_2_Cheat_Detection")

def load_and_label(path, label, limit=15):
    dfs = []
    files = glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.json"))
    print(f"📂 Found {len(files)} files in {path}, using first {limit}")
    for i, f in enumerate(files[:limit]):
        try:
            df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f)
            df["is_cheater"] = label
            dfs.append(df)
        except Exception as e:
            print("⚠️ Skipped", f, e)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_cheater = load_and_label(os.path.join(DATASET_ROOT, "with_cheater_present"), 1)
df_clean   = load_and_label(os.path.join(DATASET_ROOT, "no_cheater_present"), 0)
df = pd.concat([df_cheater, df_clean], ignore_index=True)
print(f"✅ Loaded {len(df):,} total rows.")

"""## Tidy data"""

np.random.seed(42)
df["is_cheater"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])

target_col = "is_cheater"

# Relevant columns to detecting and mitigating cheating in online games
feature_cols = [
    "velocity", "velocity_X", "velocity_Y", "velocity_Z",
    "pitch", "yaw", "is_airborne", "is_scoped", "is_alive", "X", "Y", "Z"
]
feature_cols = [c for c in feature_cols if c in df.columns]

for col in ["velocity", "pitch", "yaw"]:
    if col in df.columns:
        df[f"{col}_change"] = df[col].diff().fillna(0).abs()

agg_funcs = {
    "velocity": ["mean", "std"],
    "velocity_change": ["mean"],
    "pitch": ["mean", "std"],
    "pitch_change": ["mean"],
    "yaw": ["mean", "std"],
    "yaw_change": ["mean"],
    "is_airborne": ["mean"],
    "is_scoped": ["mean"],
    "is_alive": ["mean"]
}

# Only use existing columns
agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
agg = df.groupby("steamid").agg(agg_funcs)
agg.columns = ["_".join(c) for c in agg.columns]
agg = agg.reset_index()
#agg.head()
np.random.seed(42)
agg["is_cheater"] = np.random.choice([0,1], size=len(agg), p=[0.7,0.3])

"""## Split data into Train/Test/Validation sets"""

X = agg.drop(columns=["steamid","is_cheater"])
y = agg["is_cheater"].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

"""## Train Random Forest and Logistic Regression Classifiers"""

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)

for model, name in [(rf,"Random Forest"), (lr,"Logistic Regression")]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name}")
    print(classification_report(y_test, preds))
    print("AUC:", roc_auc_score(y_test, proba))

import matplotlib.pyplot as plt
import pandas as pd

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7,5))
imp.tail(10).plot(kind="barh")
plt.title("Top 10 Feature Importances – Random Forest")
plt.show()

"""## Summary Statistics

### Confusion Matrices
"""

fig, ax = plt.subplots(1, 2, figsize=(10,4))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=ax[0])
ax[0].set_title("Random Forest (Test)")
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=ax[1])
ax[1].set_title("Logistic Regression (Test)")
plt.show()

"""### AUC scores"""

results = pd.DataFrame({
    "Model": ["Random Forest", "Logistic Regression"],
    "Test AUC": [
        roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]),
        roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
    ]
})
results