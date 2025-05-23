> **"Intelligent Anomaly Detection in Database Security: A Triple Loop Learning Framework"**  
> 📍 *Presented at ICSC 2025 by William K.*  
> 🏛 University of Vienna, Department of Computer Science  
> 📧 a09726537@unet.univie.ac.at

---

## 🔍 Overview

The **Triple Loop Learning (TLL)** framework integrates three layers of learning to detect and adapt to anomalies in RDBMS environments:

- 🔁 **Operational loop** — anomaly detection using autoencoders trained on SQL logs.
- 🧠 **Tactical loop** — real-time reinforcement learning (DQN) for threshold/policy tuning.
- 🎯 **Strategic loop** — few-shot adaptation using meta-learning (MAML) for dynamic contexts.

It presents AI-DAC, an explainable and recursive anomaly detection framework built on a **Triple Loop Learning** architecture (supervised + RL + meta-learning), tailored for high-stakes SQL environments.

```bash

## 📌 Key Components
bash
tll-rdbms/
├── TLL_VM_Setup_Tutorial.txt     ← Full Ubuntu & WSL2 setup tutorial
├── rdbms_audit_scripts/          ← PostgreSQL auditing & forensic scripts
├── madgan_detection/             ← MAD-GAN anomaly detection pipeline
├── tll_double_loop_rl/           ← RL-only DQN policy baseline
├── tll_triple_loop_meta/         ← Final full TLL + MAML implementation
├── tll_interface/                ← Splunk, XDR, and Grafana integrations
├── tll_vm_runbook/               ← Operational runbook, test cases, stress tools
├── rl/                           ← Trained DQN policy and config
│   ├── policy_dqn.zip           ← Trained DQN model checkpoint (Stable-Baselines3)
│   ├── policy_config.json       ← Hyperparameters and training configuration for reproducibility
│   └── load_dqn_policy.py       ← Script to load and test the DQN policy on a custom Gym environment
├── models/                       ← Trained models (LSTM, RAG index)
│   └── lstm_v4.h5
├── retriever/                    ← RAG vector index + config
│   ├── bert_retriever.faiss
│   ├── retriever_config.json
│   └── build_bert_retriever.py
├── scripts/                      ← Data pre-processing & injection tools
│   ├── load_unsw.py
│   ├── inject_gan_samples.py
│   └── runbook.pdf
├── helm-chart/                   ← Helm chart for container deployment
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       └── service.yaml
├── datasets/                     ← Placeholder for logs and benchmark data
├── poster/Tll_Icsc2025_Poster.tex ← ICSC 2025 poster (LaTeX)
├── images/                       ← Diagrams, logos, and visual assets
├── LICENSE                       ← MIT License
└── README.md                     ← This file
```
