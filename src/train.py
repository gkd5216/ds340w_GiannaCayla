# Load necessary libraries
import os
import glob
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Configuring paths and parameters
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)
FILES_PER_CLASS = 50 # cheater + non-cheater files
ROWS_PER_FILE = 100000 # limits rows per file


# Utility functions
def read_partial_parquet(path, n_rows=ROWS_PER_FILE):
    table = pq.read_table(path)
    if len(table) > n_rows:
        table = table.slice(0, n_rows)
    return table.to_pandas()

def load_and_label(path, label, limit=FILES_PER_CLASS, n_rows=ROWS_PER_FILE):
    dfs = []
    files = glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.json"))
    print(f"\n📂 Found {len(files)} files in {path}, loading first {limit} files (max {n_rows:,} rows each)...")

    for i, f in enumerate(files[:limit]):
        try:
            if f.endswith(".parquet"):
                df = read_partial_parquet(f, n_rows)
            else:
                if os.path.getsize(f) < 50_000_000:
                    df = pd.read_json(f, lines=True)
                else:
                    print(f"Skipped large JSON: {f}")
                    continue
            df["is_cheater"] = label
            df["steamid"] = df["steamid"].astype(str) + "_" + os.path.basename(f).split(".")[0]
            dfs.append(df)
            print(f"Loaded {len(df):,} rows from {os.path.basename(f)} ({i+1}/{limit})")
        except Exception as e:
            print(f"Skipped {os.path.basename(f)} due to error: {e}")

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"Combined {len(combined):,} rows for label {label}")
    return combined

def feature_engineering(df):
    feature_columns = [
        "velocity", "velocity_X", "velocity_Y", "velocity_Z",
        "pitch", "yaw", "is_airborne", "is_scoped", "is_alive", "X", "Y", "Z"
        ]
    feature_columns = [c for c in feature_columns if c in df.columns]
    for col in ["velocity", "pitch", "yaw"]:
        if col in df.columns:
            df[f"{col}_change"] = df.groupby("steamid")[col].diff().fillna(0).abs()

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
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    agg = df.groupby("steamid").agg(agg_funcs)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    labels = df.groupby("steamid")["is_cheater"].max().reset_index()
    agg = agg.merge(labels, on="steamid", how="left").dropna(subset=["is_cheater"])
    print(f"Aggregated dataset shape: {agg.shape}")
    return agg

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
        )
    print("\n Class distribution (train/test):")
    print(y_train.value_counts(), "\n", y_test.value_counts())
    return X_train, X_test, y_train, y_test

# Train Random Forest and Logistic Regression Models
def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42) 
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, lr

def evaluate_models(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, digits=3)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    print(f"\n{'='*60}\n📊 {name} Results\n{'='*60}")
    print(report)
    print(f"AUC: {auc:.3f}")
    print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    out_path = os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_report.txt")
    with open(out_path, "w") as f:
        f.write(report + f"\nAUC: {auc:.3f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    return auc

def plot_feature_importance(model, X):
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 5))
    imp.tail(10).plot(kind="barh")
    plt.title("Top 10 Feature Importances – Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
    plt.close()

# Eventually implement future transformer-model
def train_transformer(X_train, y_train):
    pass

# Pipeline for all functions
if __name__ == "__main__":
    print("🚀 Starting baseline AntiCheatPT reproduction...\n")

    # --- Load and merge datasets ---
    df_cheater = load_and_label(os.path.join(DATASET_ROOT, "with_cheater_present"), 1)
    df_clean   = load_and_label(os.path.join(DATASET_ROOT, "no_cheater_present"), 0)
    df_cheater["steamid"] = "C_" + df_cheater["steamid"].astype(str)
    df_clean["steamid"]   = "N_" + df_clean["steamid"].astype(str)
    df = pd.concat([df_cheater, df_clean], ignore_index=True)
    print(f"\nTotal dataset size: {len(df):,} rows\n")

    # --- Feature Engineering ---
    agg = feature_engineering(df)
    X = agg.drop(columns=["steamid", "is_cheater"])
    y = agg["is_cheater"].astype(int)

    # --- Split ---
    X_train, X_test, y_train, y_test = split_data(X, y)

    # --- Train Baselines ---
    rf, lr = train_models(X_train, y_train)

    # --- Evaluate ---
    auc_rf = evaluate_models(rf, X_test, y_test, "Random Forest")
    auc_lr = evaluate_models(lr, X_test, y_test, "Logistic Regression")
    plot_feature_importance(rf, X)

    # --- Save Summary & Comparison ---
    summary = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "AUC": [round(auc_rf, 3), round(auc_lr, 3)]
    })
    summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n✅ Saved model summary:", summary_path)
    print(summary)

    # --- Compare with Parent Paper ---
    print("\n🎯 Parent Paper (AntiCheatPT-256) Results: Accuracy = 89.17 %, AUC = 93.36 %")
    print(f"Our Baseline (Random Forest AUC = {auc_rf:.2f}, Logistic Regression AUC = {auc_lr:.2f})")
    if auc_rf >= 0.85 or auc_lr >= 0.85:
        print("✅ Baseline achieves comparable performance to the AntiCheatPT transformer benchmark within expected margin.")
    else:
        print("⚠️ Baseline below parent paper performance — further feature engineering or transformer model recommended.")

    print("\n🏁 Implementation Platform setup complete – ready for transformer module injection.\n")