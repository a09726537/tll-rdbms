# Triple Loop Learning (TLL) for RDBMS Security - Complete Project

This package provides a complete proof-of-concept implementation of Triple Loop Learning (TLL) applied to relational database security. It includes scripts for PostgreSQL and SQL Server auditing, anomaly detection using MAD-GAN, reinforcement learning for threshold adaptation, and meta-learning for generalization across tasks.

---

## Contents

### 1. `rdbms_audit_scripts.zip`
- SQL scripts for auditing database activity in:
  - PostgreSQL (`postgresql_audit.sql`)
  - SQL Server (`sqlserver_audit.sql`)

### 2. `madgan_detection.zip`
- Python implementation of a simplified MAD-GAN model for anomaly detection in time-series or SQL log-like data.

### 3. `tll_double_loop_rl.zip`
- A Q-learning agent that dynamically adjusts anomaly detection thresholds based on feedback — representing the Double Loop of TLL.

### 4. `tll_triple_loop_meta.zip`
- A Reptile meta-learning model to simulate adaptation across different environments/tasks — the Triple Loop of TLL.

### 5. `tll_interface.zip`
- `tll_flask_api.py`: REST API to interact with anomaly detection engine.
- `tll_streamlit_ui.py`: Web interface for submitting values and visualizing anomalies.

### 6. `tll_vm_runbook.zip`
- Tutorial for setting up the full system on:
  - Ubuntu Virtual Machine
  - WSL2 on Windows

---

## Quickstart

1. **Install dependencies**: Python 3.8+, PyTorch, Flask, Streamlit
2. **Launch API**:
    ```bash
    python tll_flask_api.py
    ```
3. **Launch UI**:
    ```bash
    streamlit run tll_streamlit_ui.py
    ```
4. **Deploy and test anomaly detection models** using synthetic or real log datasets.

---

## Documentation

Full setup guide is available in:
- `TLL_VM_WSL2_Tutorial.txt` (inside `tll_vm_runbook.zip`)

---

## Conceptual Overview

This system implements the **Triple Loop Learning** framework:
- **Single Loop**: Detect anomalies (MAD-GAN)
- **Double Loop**: Learn and adjust detection thresholds (Reinforcement Learning)
- **Triple Loop**: Learn how to learn across detection tasks (Meta-Learning)

---

© 2025 – William Kandolo, University of Vienna
