#!/usr/bin/env python3
# ----------------------------------------------------------------------
#  File Name     : eval.py
#  Description   : Post-hoc evaluation script for AI-DAC Loop 1 detector
#  Author        : William K.
#  Created       : 2025-01-01
#  Last Updated  : 2025-11-22
# ----------------------------------------------------------------------
#
#  This script evaluates a trained supervised detector on a given dataset.
#  It is designed to be complementary to train.py and supports:
#    - loading an existing model artefact,
#    - loading and validating an evaluation dataset,
#    - computing metrics (AUROC, AUPRC, F1, Brier, ECE),
#    - exporting a detailed evaluation manifest and per-sample scores.
#
#  Typical usage:
#
#    python eval.py \
#        --config config/train.yaml \
#        --model models/detector_<run_id>.joblib \
#        --dataset datasets/enterprise_eval.csv \
#        --output-dir artifacts/eval_<run_id>
#
# ----------------------------------------------------------------------

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
import joblib

try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ----------------------------------------------------------------------
# Utility: Logging
# ----------------------------------------------------------------------


def log(msg: str) -> None:
    """Simple STDOUT logger with prefix."""
    print(f"[AIDAC][eval] {msg}", flush=True)


# ----------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}")

    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        if not HAS_YAML:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # type: ignore
    elif cfg_path.suffix.lower() == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {cfg_path.suffix}")


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------


def load_dataset(path: str, label_column: str) -> Dict[str, np.ndarray]:
    """
    Load evaluation dataset from CSV.

    Returns:
        {"X": features_array, "y": labels_array, "columns": feature_names}
    """
    log(f"Loading evaluation dataset from {path} ...")
    df = pd.read_csv(path)

    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataset.")

    y = df[label_column].astype(int).values
    feature_cols = [c for c in df.columns if c != label_column]
    X = df[feature_cols].values

    log(f"Loaded dataset with shape X={X.shape}, y={y.shape}")
    return {"X": X, "y": y, "columns": feature_cols}


# ----------------------------------------------------------------------
# Metrics, calibration, and reliability
# ----------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute core evaluation metrics for binary classification."""
    metrics: Dict[str, float] = {}

    if len(np.unique(y_true)) == 1:
        # Avoid exceptions when only one class present
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(int)
    metrics["f1"] = float(f1_score(y_true, y_pred))
    metrics["brier"] = float(brier_score_loss(y_true, y_score))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)

    return metrics


def reliability_diagram(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability curve: mean predicted prob vs. empirical frequency.

    Returns:
        (bin_centers, bin_true_prob)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score, bins) - 1

    bin_true_prob = np.zeros(n_bins)
    bin_pred_mean = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            bin_true_prob[i] = np.nan
            bin_pred_mean[i] = np.nan
        else:
            bin_true_prob[i] = float(np.mean(y_true[mask]))
            bin_pred_mean[i] = float(np.mean(y_score[mask]))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, bin_true_prob


def compute_ece(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) using uniform probability bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_score, bins) - 1

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(y_score[mask]))
        avg_true = float(np.mean(y_true[mask]))
        weight = float(np.sum(mask)) / float(n)
        ece += weight * abs(avg_conf - avg_true)

    return float(ece)


def export_calibration_map(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute and export calibration artefacts as JSON.
    """
    log(f"Computing calibration map (n_bins={n_bins}) ...")
    bin_centers, bin_true_prob = reliability_diagram(y_true, y_score, n_bins=n_bins)
    ece = compute_ece(y_true, y_score, n_bins=n_bins)

    calib = {
        "bin_centers": bin_centers.tolist(),
        "bin_true_prob": bin_true_prob.tolist(),
        "ece": float(ece),
        "n_bins": int(n_bins),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    log(f"Calibration map written to {out_path} (ECE={ece:.4f})")
    return calib


# ----------------------------------------------------------------------
# Main evaluation workflow
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AI-DAC supervised detector on a given dataset."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/train.yaml",
        help="Path to training configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained model file (e.g., detector_<run_id>.joblib).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to evaluation dataset (CSV).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory where evaluation artefacts will be written. "
             "Defaults to 'artifacts/eval_<run_id>'.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier (defaults to generated UUID).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Classification threshold for F1 and confusion matrix (default: 0.5).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_id = args.run_id or str(uuid.uuid4())
    log(f"Starting evaluation run_id={run_id}")

    dataset_cfg = cfg.get("dataset", {})
    label_column = dataset_cfg.get("label_column", "label")

    eval_data = load_dataset(args.dataset, label_column=label_column)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    log(f"Loading model from {model_path} ...")
    model = joblib.load(model_path)

    # Predict
    log("Running predictions on evaluation dataset ...")
    y_true = eval_data["y"]
    y_score = model.predict_proba(eval_data["X"])[:, 1]

    # Compute metrics
    metrics = compute_metrics(y_true, y_score, threshold=args.threshold)
    ece = compute_ece(y_true, y_score, n_bins=10)
    metrics["ece"] = float(ece)

    log(
        "Evaluation metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    )

    # Output directory
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("artifacts") / f"eval_{run_id}"

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Writing evaluation artefacts to {out_dir}")

    # Export calibration map
    calib_path = out_dir / "calibration_map_eval.json"
    calib = export_calibration_map(y_true, y_score, calib_path, n_bins=10)

    # Export per-sample scores
    scores_path = out_dir / "scores.csv"
    scores_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    )
    scores_df.to_csv(scores_path, index=False)
    log(f"Per-sample scores written to {scores_path}")

    # Evaluation manifest
    manifest = {
        "run_id": run_id,
        "config_path": os.path.abspath(args.config),
        "model_path": os.path.abspath(str(model_path)),
        "dataset_path": os.path.abspath(args.dataset),
        "label_column": label_column,
        "threshold": float(args.threshold),
        "metrics": metrics,
        "calibration": calib,
    }

    manifest_path = out_dir / "eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log(f"Evaluation manifest written to {manifest_path}")
    log("Evaluation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"FATAL: {exc}")
        sys.exit(1)

