# Load necessary libraries
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import random
import torch
from pathlib import Path
from torch.utils.data import Dataset
from Transformer.training.hyperparameters import train_size, test_size, val_size
import pandas as pd

class DataImporter(Dataset):
    def __init__(self, split='train', transform=None, seed=41):
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        self.samples = []
        self.transform = transform
        random.seed(seed)

        # Update paths here if needed
        cheater_dir = Path("data/full_dataset/with_cheater_present")
        not_cheater_dir = Path("data/context_windows_512/not_cheater")

        cheater_files = self._group_files_by_file_int(cheater_dir)
        non_cheater_files = self._group_files_by_file_int(not_cheater_dir)

        # Combine all keys from both classes
        all_keys = list(set(cheater_files.keys()).union(set(non_cheater_files.keys())))
        random.shuffle(all_keys)

        total = len(all_keys)
        train_end = int(train_size * total)
        val_end = train_end + int(val_size * total)

        if split == 'train':
            selected_keys = all_keys[:train_end]
        elif split == 'val':
            selected_keys = all_keys[train_end:val_end]
        else:
            selected_keys = all_keys[val_end:]

        for key in selected_keys:
            file_sources = []
            if key in cheater_files:
                file_sources.append((1, cheater_files[key]))
            if key in non_cheater_files:
                file_sources.append((0, non_cheater_files[key]))

            for label, files in file_sources:
                for file_path in files:
                    if split == 'test' and "_aug" in file_path.name:
                        continue
                    self.samples.append((file_path, label))

        # Print label distribution
        print(f"Loaded {len(self.samples)} samples for split '{split}'")
        label_counts = {0: 0, 1: 0}
        for _, label in self.samples:
            label_counts[label] += 1
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count}")

    def _group_files_by_file_int(self, directory):
        grouped = {}
        if not directory.exists():
            print(f"Directory not found: {directory}")
            return grouped
        for file in os.listdir(directory):
            if not (file.endswith('.parquet') or file.endswith('.json')):
                continue
            parts = file.replace('.parquet', '').replace('.json', '').split('-')
            file_id = None
            for p in parts:
                if p.startswith("file_"):
                    file_id = p.replace("file_", "")
                    break
                elif p.isdigit():
                    file_id = p
                    break
            if file_id is None:
                print(f"Skipping malformed filename: {file}")
                continue
            key = f"file_{file_id}"
            grouped.setdefault(key, []).append(directory / file)
        return grouped


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix == ".json":
                try:
                    df = pd.read_json(file_path, lines=True)
                except ValueError:
                    # fallback: read one line per JSON object if lengths mismatch
                    df = pd.read_json(file_path, lines=False)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            # keep only numeric data
            df = df.select_dtypes(include=["number"]).fillna(0)
            if df.empty:
                raise ValueError("Empty numeric DataFrame")

            data = torch.tensor(df.values, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            if self.transform:
                data = self.transform(data)
            return data, label

        except Exception as e:
            print(f"⚠️ Skipped file {file_path.name} due to error: {e}")
            # return a small zero tensor instead of crashing
            return torch.zeros((1, 1)), torch.tensor(label, dtype=torch.float32)

    
if __name__ == "__main__":
    print("🔍 Testing dataset splits...")
    for split in ["train", "val", "test"]:
        ds = DataImporter(split=split)
        print(f"Split: {split}, total samples = {len(ds)}")
        labels = [label.item() if hasattr(label, "item") else label for _, label in ds]
        unique, counts = pd.Series(labels).value_counts().index, pd.Series(labels).value_counts().values
        print("Label distribution:")
        for u, c in zip(unique, counts):
            print(f"  Label {u}: {c}")
        print("-" * 40)