# Author: William Kandolo
import torch
import pandas as pd
from preprocess import normalize_sql, vectorize_sequence, token_to_idx
from madgan_train import compute_anomaly_score
from generator import Generator
from discriminator import Discriminator

# Parameters
SEQ_LEN = 20
INPUT_DIM = 1  # since we use one-hot or index only
HIDDEN_DIM = 64

# Load models
generator = Generator(INPUT_DIM, HIDDEN_DIM, SEQ_LEN)
discriminator = Discriminator(INPUT_DIM, HIDDEN_DIM)
generator.load_state_dict(torch.load("models/generator.pth"))
discriminator.load_state_dict(torch.load("models/discriminator.pth"))
generator.eval()
discriminator.eval()

# Read new queries
df = pd.read_csv("data/queries.csv")
alerts = []

for query in df["query"]:
    tokens = normalize_sql(query)
    idx_seq = vectorize_sequence(tokens, max_len=SEQ_LEN)
    tensor_seq = torch.tensor(idx_seq).unsqueeze(0).unsqueeze(-1).float()  # shape: (1, seq_len, 1)

    score = compute_anomaly_score(generator, discriminator, tensor_seq).item()
    if score > 0.8:
        alerts.append((query, score))

# Write alerts to log
if alerts:
    with open("alerts/anomalies.log", "a") as f:
        for query, score in alerts:
            f.write(f"ANOMALY DETECTED: {query} | Score: {score:.4f}\n")
    print(f"{len(alerts)} anomalies detected. Logged to alerts/anomalies.log")
else:
    print("No anomalies detected.")
