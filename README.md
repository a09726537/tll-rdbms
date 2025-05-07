# TLL-RDBMS: Triple Loop Learning for Intelligent Anomaly Detection in Databases

This repository contains the full implementation, poster, and datasets for the research presented in:

**"Intelligent Anomaly Detection in Database Security: A Triple Loop Learning Framework"**  
📍 *Presented at ICSC 2025 by William Kandolo*  
🏛 University of Vienna, Austria

---

## 🔍 Overview

The **Triple Loop Learning (TLL)** framework integrates three layers of learning to detect and respond to anomalies in RDBMS environments:

- 🔁 **Operational loop** – Anomaly detection using autoencoders trained on SQL logs.
- 🧠 **Tactical loop** – Reinforcement Learning (DQN) for real-time threshold and policy updates.
- 🎯 **Strategic loop** – Meta-learning (MAML) for few-shot adaptation across database contexts.

Together, these components form the **AI-Driven Adaptive Control (AI-DAC)** system, enhanced by a Retrieval-Augmented Generation (RAG) module for explainability.

---

## 📁 Repository Structure

```bash
tll-rdbms/
├── README.md                      ← This file
├── TLL_VM_Setup_Tutorial.txt     ← Full VM & WSL2 setup guide
├── rdbms_audit_scripts/          ← PostgreSQL audit & forensic tools
├── madgan_detection/             ← MAD-GAN training & detection module
├── tll_double_loop_rl/           ← DQN-only baseline implementation
├── tll_triple_loop_meta/         ← Final TLL implementation with MAML
├── tll_interface/                ← Splunk/XDR dashboard integration
├── tll_vm_runbook/               ← Operational runbook & test cases
├── poster/Tll_Icsc2025_Poster.tex← ICSC 2025 poster (LaTeX source)
├── images/                       ← Logos and experiment figures
├── datasets/                     ← Placeholder for UNSW-NB15, logs, MAD-GAN outputs
└── LICENSE                       ← MIT or academic license

## 🚀 Setup Instructions

To get started with the TLL-RDBMS project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/a09726537/tll-rdbms.git
cd tll-rdbms

### 2. Install Python Dependencies
pip install -r requirements.txt
Or install manually:
pip install torch tensorflow scikit-learn faiss-cpu matplotlib pandas

### 3. Follow VM Setup Guide

Open the file: TLL_VM_Setup_Tutorial.txt

Use WSL2, Ubuntu VM, or local Linux environment

Configure PostgreSQL and datasets as instructed
1. madgan_detection/           → Generate synthetic anomalies
2. tll_triple_loop_meta/       → Train and evaluate full TLL model
3. tll_interface/              → Visualize outputs, integrate with Splunk

### 4. Optional Testing & Validation

Use the test cases in tll_vm_runbook/

Validate outputs against labeled SQL logs and UNSW-NB15


@misc{tllgithub2025,
  author = {William Kandolo},
  title = {Triple Loop Learning for RDBMS Security (TLL-RDBMS)},
  year = {2025},
  howpublished = {\url{https://github.com/a09726537/tll-rdbms}},
  note = {Presented at ICSC 2025, University of Vienna}
}

## 📜 License

This project is provided under an open-source academic license.  
See [LICENSE](./LICENSE) for full terms.
---



