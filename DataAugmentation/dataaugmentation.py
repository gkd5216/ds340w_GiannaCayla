import pandas as pd
import os
import numpy as np

folder_dir = os.path.join("data", "no_cheater_present")

files = os.listdir(folder_dir)
original_file_count = len(files)

axes = ["X", "Y", "Z"]
aug_amount = 1
noise_std = 0.01

skipped_files = []

for idx, file in enumerate(files):
    if not file.endswith(".parquet"):
        continue

    file_path = os.path.join(folder_dir, file)
    print(f"\rProcessing file {idx + 1}/{original_file_count}: {file}", end="", flush=True)

    df = pd.read_parquet(file_path)
    required_cols = [f"attacker_{a}" for a in axes] + [f"victim_{a}" for a in axes]
    if not all(col in df.columns for col in required_cols):
        skipped_files.append(file)
        continue  

    for aug_idx in range(aug_amount):
        df_aug = df.copy()

        for axis in axes:
            attacker_col = f"attacker_{axis}"
            victim_col = f"victim_{axis}"

            noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df))
            df_aug[attacker_col] = np.clip(df_aug[attacker_col] + noise, 0.0, 1.0)
            df_aug[victim_col] = np.clip(df_aug[victim_col] + noise, 0.0, 1.0)

        new_filename = file.replace(".parquet", f"_aug{aug_idx}.parquet")
        new_path = os.path.join(folder_dir, new_filename)
        df_aug.astype(np.float32).to_parquet(new_path, index=False)

final_file_count = len(os.listdir(folder_dir))
created_files = final_file_count - original_file_count

print(f"\n {created_files} new files created.")
if skipped_files:
    print(f"Skipped {len(skipped_files)} files due to missing columns:")
    for f in skipped_files[:10]: 
        print(f" - {f}")
