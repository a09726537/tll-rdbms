# 🧠 TLL-RDBMS: Triple Loop Learning for Intelligent Anomaly Detection in Databases

This repository contains the full implementation, datasets, and poster for the research presented in:

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

These components form the **AI-DAC (AI-Driven Adaptive Control)** system, enhanced with a **Retrieval-Augmented Generation (RAG)** module for natural-language explanations and analyst trust.

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
├── scripts/                      ← Data preparation and anomaly injection tools
│   ├── load_unsw.py              ← Converts UNSW-NB15 to structured SQL logs
│   ├── inject_gan_samples.py     ← Injects GAN-generated anomalies
│   └── runbook.pdf               ← Experimental runbook (PDF)
├── poster/Tll_Icsc2025_Poster.tex ← ICSC 2025 poster (LaTeX source)
├── images/                       ← Figures, diagrams, logos
├── datasets/                     ← Placeholder for SQL logs, UNSW-NB15, MAD-GAN samples
└── LICENSE                       ← MIT or academic license
