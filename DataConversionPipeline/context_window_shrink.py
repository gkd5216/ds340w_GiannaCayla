import os
import pandas as pd
import numpy as np

# Define input/output directories
base_input_dir = os.path.join("data")
base_output_dir = os.path.join("data_shrunk")

subfolders = ["with_cheater_present", "no_cheater_present"]

# Context window cropping
start_idx = 224
end_idx = 480  # not inclusive

for folder in subfolders:
    input_dir = os.path.join(base_input_dir, folder)
    output_dir = os.path.join(base_output_dir, folder)
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]

    for idx, file in enumerate(files):
        input_path = os.path.join(input_dir, file)
        df = pd.read_parquet(input_path)

        if len(df) < end_idx:
            print(f"⚠️ Skipping {file}: too short ({len(df)} rows)")
            continue

        df_shrunk = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # ✅ Cast only numeric (float/int) columns
        for col in df_shrunk.columns:
            if pd.api.types.is_numeric_dtype(df_shrunk[col]):
                try:
                    df_shrunk[col] = df_shrunk[col].astype(np.float32)
                except Exception as e:
                    print(f"⚠️ Could not convert column {col} in {file}: {e}")

        output_path = os.path.join(output_dir, file)
        df_shrunk.to_parquet(output_path, index=False)

        print(f"[{folder}] {idx+1}/{len(files)} → {file} cropped and saved.")

print("\n✅ All valid .parquet files have been safely shrunk.")
