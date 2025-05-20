# tll_flask_api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import shap
import json
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# === Load Models ===
autoencoder = load_model("models/supervised/lstm_v4.h5")
scaler = joblib.load("models/supervised/scaler.pkl")
explainer = joblib.load("explainer/shap_explainer.pkl")  # Precomputed SHAP explainer

# Default threshold (can be dynamically updated)
THRESHOLD = 0.7


@app.route("/detect", methods=["POST"])
def detect():
    """
    Detect anomalies from incoming SQL log records.
    Payload: JSON list of records with numerical features.
    Returns: Prediction and explanation
    """
    global THRESHOLD

    try:
        data = request.get_json()
        records = pd.DataFrame(data["features"])  # expects list of dicts or list of lists

        # Normalize input
        X_scaled = scaler.transform(records)

        # Predict using autoencoder
        X_pred = autoencoder.predict(X_scaled)
        loss = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        is_anomaly = loss > THRESHOLD

        # SHAP explanation (optional)
        shap_values = explainer.shap_values(X_scaled)
        shap_summary = np.mean(np.abs(shap_values), axis=0).tolist()

        return jsonify({
            "anomalies": is_anomaly.tolist(),
            "reconstruction_loss": loss.tolist(),
            "explanation": shap_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/update_threshold", methods=["POST"])
def update_threshold():
    """
    Updates the anomaly detection threshold at runtime.
    Payload: { "threshold": float }
    """
    global THRESHOLD
    data = request.get_json()
    try:
        new_threshold = float(data["threshold"])
        THRESHOLD = new_threshold
        return jsonify({"message": f"Threshold updated to {THRESHOLD}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_loaded": True, "threshold": THRESHOLD})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
