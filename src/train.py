# Load necessary libraries
import os
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Configuring paths and parameters
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FILES_PER_CLASS = 15 # number of cheater and non-cheater files
ROWS_PER_FILE = 200000


# Load dataset
def read_partial_parquet(path, n_rows=200000):
    """Read only a portion of a large parquet file to save memory."""
    table = pq.read_table(path)
    if len(table) > n_rows:
        table = table.slice(0, n_rows)
    return table.to_pandas()

def load_and_label(path, label, limit=15, n_rows=200000):
    """Load up to `limit` files, partially, and attach a label."""
    dfs = []
    files = glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.json"))
    print(f"\n📂 Found {len(files)} files in {path}, loading first {limit} files (max {n_rows:,} rows each)...")

    for i, f in enumerate(files[:limit]):
        try:
            if f.endswith(".parquet"):
                df = read_partial_parquet(f, n_rows)
            else:
                # only read small JSONs
                if os.path.getsize(f) < 50_000_000:
                    df = pd.read_json(f, lines=True)
                else:
                    print(f"⚠️ Skipped large JSON: {f}")
                    continue
            df["is_cheater"] = label
            dfs.append(df)
            print(f"  ✅ Loaded {len(df):,} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"⚠️ Skipped {os.path.basename(f)} due to error: {e}")
    
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"Combined {len(combined):,} rows for label {label}")
    return combined

df_cheater = load_and_label(os.path.join(DATASET_ROOT, "with_cheater_present"), 1)
df_clean   = load_and_label(os.path.join(DATASET_ROOT, "no_cheater_present"), 0)
df = pd.concat([df_cheater, df_clean], ignore_index=True)
print(f"\n Total dataset size: {len(df):,} rows\n")

## Tidy dataframe and feature engineering
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
labels = df.groupby("steamid")["is_cheater"].max().reset_index()
agg = agg.merge(labels, on="steamid", how="left").dropna(subset=["is_cheater"])
print(f"Aggregated dataset shape: {agg.shape}")

## Split data into Train/Test/Validation sets
X = agg.drop(columns=["steamid","is_cheater"])
y = agg["is_cheater"].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\n🔍 y_train class distribution:")
print(y_train.value_counts())
print("\n🔍 y_test class distribution:")
print(y_test.value_counts())

if y_train.nunique() < 2:
    raise ValueError("❌ Only one class in y_train! Model training requires at least two classes. Check your data and stratification.")
if y_test.nunique() < 2:
    print("⚠️ Only one class in y_test. Model evaluation will be limited.")

## Train Random Forest and Logistic Regression Models

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

## Evaluation and Visualization
def evaluate_model(model, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, digits=3)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name} Results\n{'='*40}\n{report}\nAUC: {auc:.3f}")
    with open(os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_report.txt"), "w") as f:
        f.write(report + f"\nAUC: {auc:.3f}\n")
    return auc

auc_rf = evaluate_model(rf, "Random Forest")
auc_lr = evaluate_model(lr, "Logistic Regression")

## Feature Importance

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
imp.tail(10).plot(kind="barh")
plt.title("Top 10 Feature Importances – Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
plt.close()