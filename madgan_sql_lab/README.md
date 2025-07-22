<!-- Author: William Kandolo -->
# MAD-GAN SQL Anomaly Detection Lab

This lab demonstrates an unsupervised GAN-based approach for detecting anomalies in SQL Server logs using LSTM-based Generator and Discriminator models.

---

## 🔧 Setup Instructions

### 1. Create a Python virtual environment
```bash
python3 -m venv madgan_env
source madgan_env/bin/activate
```

### 2. Install required packages
```bash
pip install torch numpy pandas scikit-learn sqlparse matplotlib jupyter
```

### 3. Train the MAD-GAN model
```bash
python3 madgan_train.py
```

Trained models will be saved to the `models/` folder as:
- `generator.pth`
- `discriminator.pth`

### 4. Detect anomalies in new SQL queries
```bash
python3 detect_anomalies.py
```

If anomalies are detected, they will be logged in `alerts/anomalies.log`.

---

## 📁 Project Structure and File Explanations

| File / Folder           | Description |
|-------------------------|-------------|
| `generator.py`          | Defines the LSTM-based Generator model for sequence generation. |
| `discriminator.py`      | Defines the LSTM-based Discriminator model to classify real vs. fake sequences. |
| `preprocess.py`         | Contains functions to normalize SQL queries and vectorize them into fixed-length token sequences. |
| `madgan_train.py`       | Trains the MAD-GAN model using input vectors. Saves trained models to `models/`. |
| `detect_anomalies.py`   | Loads new queries, processes them, scores them with the trained GAN, and logs anomalies. |
| `data/queries.csv`      | Sample SQL queries for training or testing. Replace or update this file with your real SQL logs. |
| `models/`               | Directory for storing trained model files (`.pth`) after training. |
| `alerts/anomalies.log`  | Output file where detected anomalies are logged, including their query and anomaly score. |
| `README.md`             | This instruction and documentation file. |

---

## 📌 Notes

- You can integrate `detect_anomalies.py` with `cron` for scheduled detection.
- You can add Slack/email/webhook alerting inside `detect_anomalies.py`.

---

For support or integration help, contact the developer or use Jupyter notebooks to explore the models step-by-step.
