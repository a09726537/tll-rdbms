# TLL-RDBMS: Triple Loop Learning for Intelligent Anomaly Detection in Databases

This repository contains the full implementation, poster, and datasets for the research presented in:

**"Intelligent Anomaly Detection in Database Security: A Triple Loop Learning Framework"**  
📍 *Presented at ICSC 2025 by Dr. William Kandolo*  
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
