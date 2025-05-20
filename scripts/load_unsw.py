
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# === Configuration ===
DATA_DIR = "./data"
RAW_FILE = "UNSW-NB15.csv"
OUTPUT_FILE = "unsw_processed.csv"
LABEL_COLUMN = "label"
DROP_COLUMNS = ['id', 'attack_cat', 'proto', 'service', 'state']  # Modify based on feature engineering

def load_dataset(path):
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess(df):
    # Drop non-informative or categorical features (if not embedding them)
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')

    # Encode label
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # Normalize features
    features = df.drop(columns=[LABEL_COLUMN])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # Add label back
    df_scaled[LABEL_COLUMN] = df[LABEL_COLUMN].values
    print("Preprocessing complete.")
    return df_scaled

def save_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Processed dataset saved to {path}")

if __name__ == "__main__":
    input_path = os.path.join(DATA_DIR, RAW_FILE)
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)

    df_raw = load_dataset(input_path)
    df_processed = preprocess(df_raw)
    save_dataset(df_processed, output_path)
