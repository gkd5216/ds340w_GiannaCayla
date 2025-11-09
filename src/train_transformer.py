"""
train_transformer.py – DS340W Implementation Platform (Parent Paper Reproduction)
Author: Gianna DeLorenzo & Cayla Stevenson
Parent Paper: "AntiCheatPT: A Transformer-Based Approach to Cheat Detection in Competitive Computer Games"
Dataset: CS2CD.Counter-Strike_2_Cheat_Detection
Goal: Reproduce results similar to AntiCheatPT-256 (Accuracy ≈ 0.89, AUC ≈ 0.93)
"""

# ======================
# Imports
# ======================
import os
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CONTEXT_WINDOW = 256
INPUT_DIM = 6                 # features per tick: X, Y, Z, velocity, pitch, yaw
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 4
THRESHOLD = 0.7

# Transformer Model
class AntiCheatPT256(nn.Module):
    """Simplified reproduction of AntiCheatPT-256 (Transformer encoder)."""
    def __init__(self, input_dim=INPUT_DIM, d_model=32, nhead=1, num_layers=4, dim_ff=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_proj(x)
        x = self.encoder(x)
        cls_token = x[:, 0, :]  # first tick as sequence representation
        out = self.classifier(cls_token)
        return out.squeeze(-1)

# Data Loading (Local)
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
cheater_path = os.path.join(DATASET_DIR, "with_cheater_present")
clean_path   = os.path.join(DATASET_DIR, "no_cheater_present")

def read_parquet_subset(folder, limit=5, n_rows=100000):
    """Read Parquet files and keep numeric + steamid columns."""
    files = glob.glob(os.path.join(folder, "*.parquet"))
    dfs = []
    for f in files[:limit]:
        try:
            table = pq.read_table(f)
            df = table.to_pandas()
            numeric_df = df.select_dtypes(include=["number", "bool"]).copy()
            if "steamid" in df.columns:
                numeric_df["steamid"] = df["steamid"].astype(str)
            df = numeric_df
            keep = [c for c in df.columns if c in [
                "steamid", "tick", "X", "Y", "Z",
                "velocity", "pitch", "yaw",
                "is_airborne", "is_scoped", "is_alive"
            ]]
            df = df[keep]
            df.loc[:, "is_cheater"] = 1 if "with_cheater_present" in folder else 0
            dfs.append(df.head(n_rows))
            print(f"✅ Loaded {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"⚠️ Skipped {os.path.basename(f)}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print("📦 Loading local CS2CD subset...")
df_cheater = read_parquet_subset(cheater_path)
df_clean   = read_parquet_subset(clean_path)
ds = pd.concat([df_cheater, df_clean], ignore_index=True)
print(f"✅ Loaded total {len(ds):,} rows")

# Context Window Creation
def create_context_windows(df, window_size=CONTEXT_WINDOW):
    """Create sequential context windows per player."""
    windows, labels = [], []
    grouped = df.groupby("steamid")
    for _, player_df in grouped:
        arr = player_df[["X", "Y", "Z", "velocity", "pitch", "yaw"]].values
        label = int(player_df["is_cheater"].iloc[0])
        if len(arr) >= window_size:
            for i in range(0, len(arr) - window_size, window_size):
                win = arr[i:i + window_size]
                windows.append(win)
                labels.append(label)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.float32)

print("✅ Generating context windows...")
X, y = create_context_windows(ds)
print(f"✅ Generated {len(X)} windows with shape {X.shape[1:]}")

# Clean + Normalize Data
print("🧹 Cleaning data (removing NaN/Inf and normalizing)...")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_min = X.min(axis=(0, 1), keepdims=True)
X_max = X.max(axis=(0, 1), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-8)
X = np.clip(X, 0, 1)

# Split into Tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
total_size = len(X_tensor)
train_size = int(0.7 * total_size)
val_size   = int(0.15 * total_size)
test_size  = total_size - train_size - val_size

train_ds, val_ds, test_ds = random_split(X_tensor, [train_size, val_size, test_size])

train_loader = DataLoader(TensorDataset(X_tensor[train_ds.indices], y_tensor[train_ds.indices]), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_tensor[val_ds.indices], y_tensor[val_ds.indices]), batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_tensor[test_ds.indices], y_tensor[test_ds.indices]), batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
model = AntiCheatPT256(input_dim=INPUT_DIM).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training Loop
print("\n🚀 Training AntiCheatPT Transformer...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} - Avg Train Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds)

print("\n📊 Evaluation Results (AntiCheatPT-256 Reproduction):")
print(f"Accuracy:  {acc*100:.2f}%")
print(f"AUC:       {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# Save Results
result_path = os.path.join(RESULTS_DIR, "transformer_results.txt")
with open(result_path, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")
print(f"\n✅ Results saved to {result_path}")

# Compare with Parent Paper
print("\n🎯 Parent Paper (AntiCheatPT-256) Target: Accuracy = 89.17%, AUC = 0.9336")
print(f"Your Model Results: Accuracy = {acc*100:.2f}%, AUC = {auc:.4f}")
if auc >= 0.90:
    print("✅ Achieved performance comparable to parent paper baseline.")
else:
    print("⚠️ Slightly below parent paper performance – consider training on more files or more epochs.")

print("\n🏁 Transformer reproduction complete.\n")

