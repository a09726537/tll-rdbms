import os
import pandas as pd
import numpy as np

# === Configuration ===
REAL_LOG_PATH = "./data/unsw_processed.csv"
GAN_SYNTHETIC_PATH = "./data/madgan_generated.csv"
OUTPUT_PATH = "./data/combined_logs.csv"
ANOMALY_LABEL = 1
NORMAL_LABEL = 0

def load_data(real_path, gan_path):
    print("[INFO] Loading real logs...")
    real_df = pd.read_csv(real_path)
    print(f"Real log shape: {real_df.shape}")

    print("[INFO] Loading GAN-generated logs...")
    gan_df = pd.read_csv(gan_path)
    print(f"GAN log shape: {gan_df.shape}")

    return real_df, gan_df

def tag_data(real_df, gan_df):
    # Ensure both datasets have the same features
    common_cols = list(set(real_df.columns) & set(gan_df.columns))
    real_df = real_df[common_cols]
    gan_df = gan_df[common_cols]

    # Tag GAN samples as anomalies
    if 'label' in gan_df.columns:
        gan_df['label'] = ANOMALY_LABEL
    else:
        gan_df.insert(len(gan_df.columns), 'label', ANOMALY_LABEL)

    # Ensure real samples are marked as normal
    real_df['label'] = NORMAL_LABEL

    return real_df, gan_df

def merge_and_shuffle(real_df, gan_df):
    combined_df = pd.concat([real_df, gan_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    print(f"[INFO] Combined dataset shape: {combined_df.shape}")
    return combined_df

def save_output(df, path):
    df.to_csv(path, index=False)
    print(f"[INFO] Output written to: {path}")

if __name__ == "__main__":
    real_df, gan_df = load_data(REAL_LOG_PATH, GAN_SYNTHETIC_PATH)
    real_df, gan_df = tag_data(real_df, gan_df)
    combined_df = merge_and_shuffle(real_df, gan_df)
    save_output(combined_df, OUTPUT_PATH)
