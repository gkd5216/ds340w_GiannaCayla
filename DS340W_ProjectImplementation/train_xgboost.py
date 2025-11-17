# XGBoost Enhancement for DS340W Project
import os
import pandas as pd
import pyarrow.parquet as pq
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Paths
DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHEATER_PATH = os.path.join(DATASET_ROOT, "data", "full_dataset", "with_cheater_present")
NON_CHEATER_PATH = os.path.join(DATASET_ROOT, "data", "context_windows_512", "not_cheater")
RESULTS_DIR = os.path.join(DATASET_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FILES_PER_CLASS = 50
ROWS_PER_FILE = 100_000

def read_partial_parquet(path, n_rows=ROWS_PER_FILE):
    table = pq.read_table(path)
    if len(table) > n_rows:
        table = table.slice(0, n_rows)
    return table.to_pandas()

def load_and_label(path, label, limit=FILES_PER_CLASS, n_rows=ROWS_PER_FILE):
    dfs = []
    files = glob.glob(os.path.join(path, "*.parquet"))
    print(f"Found {len(files)} files in {path}, loading first {limit} files...")
    for i, f in enumerate(files[:limit]):
        try:
            df = read_partial_parquet(f, n_rows)
            df["is_cheater"] = label
            df["steamid"] = os.path.basename(f).split(".")[0]
            dfs.append(df)
            print(f"Loaded {len(df):,} rows from {os.path.basename(f)} ({i+1}/{limit})")
        except Exception as e:
            print(f"Skipped {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def feature_engineering(df):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "is_cheater"]
    agg_funcs = {col: ["mean", "std"] for col in numeric_cols}
    agg = df.groupby("steamid").agg(agg_funcs)
    agg.columns = ["_".join(c) for c in agg.columns]
    labels = df.groupby("steamid")["is_cheater"].max().reset_index()
    agg = agg.merge(labels, on="steamid", how="left")
    return agg

if __name__ == "__main__":
    print("Training XGBoost Model...")

    # Load both datasets
    df_cheater = load_and_label(CHEATER_PATH, 1)
    df_clean = load_and_label(NON_CHEATER_PATH, 0)
    df = pd.concat([df_cheater, df_clean], ignore_index=True)

    # Balance dataset
    print("Before balancing:", df["is_cheater"].value_counts().to_dict())
    min_count = df["is_cheater"].value_counts().min()
    df = df.groupby("is_cheater", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    print("After balancing:", df["is_cheater"].value_counts().to_dict())

    agg = feature_engineering(df)
    X = agg.drop(columns=["steamid", "is_cheater"]).fillna(0)
    y = agg["is_cheater"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, preds, digits=3)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    print("\nXGBoost Results:")
    print(report)
    print(f"AUC: {auc:.3f}")
    print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    # Save results
    summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
    summary = pd.DataFrame([{
        "Model": "XGBoost",
        "AUC": round(auc, 3)
    }])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        summary = pd.concat([existing, summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\n Updated model_summary.csv with XGBoost results:\n{summary}")

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, 
                y=importances.index, 
                palette="viridis",
                hue = importances.index,
                legend=False)
    plt.title("Top 10 Feature Importances – XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "xgboost_feature_importance.png"), dpi=300)
    plt.show()
