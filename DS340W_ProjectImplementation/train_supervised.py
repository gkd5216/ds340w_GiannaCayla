# Load necessary libraries
import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

# Configuration
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHEATER_PATH = os.path.join(DATASET_ROOT, "data", "full_dataset", "with_cheater_present")
NON_CHEATER_PATH = os.path.join(DATASET_ROOT, "data", "context_windows_512", "not_cheater")
RESULTS_DIR = os.path.join(DATASET_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES_PER_CLASS = 10
ROWS_PER_FILE = 20000

# Utility Functions
def read_partial_parquet(path, n_rows=ROWS_PER_FILE):
    table = pq.read_table(path)
    if len(table) > n_rows:
        table = table.slice(0, n_rows)
    return table.to_pandas()


def load_and_label(path, label, limit=FILES_PER_CLASS, n_rows=ROWS_PER_FILE):
    dfs = []
    files = glob.glob(os.path.join(path, "*.parquet"))
    for i, f in enumerate(files[:limit]):
        try:
            df = read_partial_parquet(f, n_rows)
            df["is_cheater"] = label
            df["steamid"] = os.path.basename(f).split(".")[0]
            dfs.append(df)
        except Exception as e:
            pass

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined

def train_models(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=150, 
                                max_depth=6, 
                                class_weight="balanced", 
                                random_state=42)
    lr = LogisticRegression(max_iter=3000, 
                            class_weight="balanced")
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    return rf, lr

def feature_engineering(df):
    # Ensure a 'steamid' column exists
    if "steamid" not in df.columns:
        print("No 'steamid' column found — assigning unique IDs per row group.")
        df["steamid"] = [f"id_{i}" for i in range(len(df))]

    # Identify numeric columns (excluding the label)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "is_cheater"]
    LEAKY_COLUMNS = {"is_cheater", "cheater_present", "label", "cheat_score"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in LEAKY_COLUMNS]
    numeric_cols = [c for c in numeric_cols if "cheater" not in c.lower()]
    if not numeric_cols:
        raise ValueError("No numeric columns found for aggregation. Please verify dataset structure.")
    else:
        pass

    # Adds change features
    for col in ["velocity", "pitch", "yaw", "player_velocity_x", "aim_pitch", "aim_yaw"]:
        if col in df.columns:
            df[f"{col}_change"] = df.groupby("steamid")[col].diff().fillna(0).abs()
            numeric_cols.append(f"{col}_change")

    # Aggregation
    agg_funcs = {col: "mean" for col in numeric_cols}
    agg = df.groupby("steamid").agg(agg_funcs)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Merge back labels
    labels = df.groupby("steamid")["is_cheater"].max().reset_index()
    agg = agg.merge(labels, on="steamid", how="left").dropna(subset=["is_cheater"])
    return agg

def split_data(X, y, groups):
    splitter1 = GroupShuffleSplit(test_size=0.30, n_splits=1, random_state=42)
    train_val_idx, test_idx = next(splitter1.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    splitter2 = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    train_idx, val_idx = next(splitter2.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]

    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_models(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, preds, digits=3)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    accuracy = accuracy_score(y_test, preds)

    print(f"\n {name} Results")
    print(report)
    print(f"AUC: {auc:.3f} | Accuracy: {accuracy:.3f}")
    print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    with open(os.path.join(RESULTS_DIR, f"{name.lower().replace(' ', '_')}_report.txt"), "w") as f:
        f.write(report + f"\nAUC: {auc:.3f}\nAccuracy: {accuracy:.3f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_string())

    return auc, accuracy


def plot_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        plt.figure(figsize=(7, 5))
        imp.tail(10).plot(kind="barh")
        plt.title("Top 10 Feature Importances – Random Forest")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()

# Pipeline
if __name__ == "__main__":
    df_cheater = load_and_label(CHEATER_PATH, 1)
    df_clean = load_and_label(NON_CHEATER_PATH, 0)

    if "steamid" not in df_cheater.columns:
        df_cheater["steamid"] = [f"C_{i}" for i in range(len(df_cheater))]
    if "steamid" not in df_clean.columns:
        df_clean["steamid"] = [f"N_{i}" for i in range(len(df_clean))]
    df = pd.concat([df_cheater, df_clean], ignore_index=True)
    print(f"\n Total dataset size: {len(df):,} rows\n")

    min_count = df["is_cheater"].value_counts().min()
    df = (
        df.groupby("is_cheater", group_keys=False)
        .apply(lambda x: x.sample(min_count, random_state=42))
        .reset_index(drop=True)
        )
    df = df.sample(frac=0.5, random_state=42) #Random subsample

    agg = feature_engineering(df)
    groups = agg["steamid"]
    X = agg.drop(columns=["steamid", "is_cheater"])
    y = agg["is_cheater"].astype(int)
    X = X.fillna(0)
    print(f"Missing values in X: {X.isna().sum().sum()}")

    X = X + np.random.normal(0, 0.05, X.shape)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, groups)
    rf, lr = train_models(X_train, y_train)

    auc_rf, acc_rf = evaluate_models(rf, X_test, y_test, "Random Forest")
    auc_lr, acc_lr = evaluate_models(lr, X_test, y_test, "Logistic Regression")

    plot_feature_importance(rf, X)

    summary = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "AUC": [round(auc_rf, 3), round(auc_lr, 3)],
        "Accuracy": [round(acc_rf, 3), round(acc_lr, 3)]
    })
    summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
    summary.to_csv(summary_path, index=False)

    rf_path = os.path.join(RESULTS_DIR, "random_forest_model.pkl")
    lr_path = os.path.join(RESULTS_DIR, "logistic_regression_model.pkl")
    X_test_path = os.path.join(RESULTS_DIR, "X_test.csv")
    y_test_path = os.path.join(RESULTS_DIR, "y_test.csv")
    X_val_path = os.path.join(RESULTS_DIR, "X_val.csv")
    y_val_path = os.path.join(RESULTS_DIR, "y_val.csv")

    joblib.dump(rf, rf_path)
    joblib.dump(lr, lr_path)

    rf_val_preds = rf.predict(X_val)
    lr_val_preds = lr.predict(X_val)

    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    X_val.to_csv(X_val_path, index=False)
    y_val.to_csv(y_val_path, index=False)

    print(f"  • {rf_path}")
    print(f"  • {lr_path}")
    print(f"  • {X_test_path}")
    print(f"  • {y_test_path}")
    print("\n Saved model summary:", summary_path)

    print(f"Our Results: Random Forest AUC = {auc_rf:.2f}, Accuracy = {acc_rf:.2f}")
    print(f"             Logistic Regression AUC = {auc_lr:.2f}, Accuracy = {acc_lr:.2f}")
