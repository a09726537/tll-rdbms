# TLL-RDBMS  
### Triple Loop Learning Framework for RDBMS Cybersecurity  
**Author: W. Kandolo — University of Vienna**

---

## 📌 Overview

**TLL-RDBMS** is a modular, research-grade cybersecurity framework designed to protect **Relational Database Management Systems (RDBMS)** using advanced Artificial Intelligence methods, including:

- **Triple Loop Learning (TLL)**  
- **Reinforcement Learning (DQN)**  
- **Meta-Learning (MAML)**  
- **Generative Adversarial Networks (MAD-GAN)**  
- **Robustness Testing & Latency Benchmarking**  
- **Production Deployment (Docker + Kubernetes)**

The goal is to create **autonomous, adaptive, and explainable security** for production-grade SQL environments.

---

## 📁 Repository Structure

tll-rdbms/
├── scripts/
│ ├── bootstrap.sh # VM/Environment bootstrap
│ ├── vm-init.sh # Ubuntu/WSL2 initialization
│ ├── train.py # Main supervised anomaly training (Loop 1)
│ ├── train_rl.py # Reinforcement Learning agent (Loop 2)
│ ├── meta_adapt.py # Meta-learning (MAML) engine (Loop 3)
│ └── eval.py # Unified evaluation pipeline
├── docker/
│ └── Dockerfile # Production-ready Docker build
├── k8s/
│ ├── deployment.yaml # Kubernetes Deployment
│ ├── service.yaml # Service (ClusterIP/LoadBalancer)
│ ├── configmap.yaml # Environment configuration
│ └── hpa.yaml # Horizontal Pod Autoscaler
├── tests/
│ ├── test_pipeline.py # End-to-end system test
│ ├── robustness.py # Noise, feature drop, FGSM robustness
│ └── test_utils.py # Assertions, timing, synthetic datasets
└── bench/
└── latency_test.py # Benchmark CPU/GPU inference latency

---

## Features

### Triple Loop Learning (TLL)
| Loop | Component | Description |
|------|-----------|-------------|
| **Loop 1** | `train.py` | Supervised anomaly detection (baseline learning) |
| **Loop 2** | `train_rl.py` | RL-based dynamic response & control (DQN) |
| **Loop 3** | `meta_adapt.py` | MAML-style adaptation to new workloads |

### MAD-GAN for Anomaly Simulation
- Synthetic adversarial sequence generation  
- GAN-based stress testing  
- Enhances classifier robustness  

### Meta-Learning (MAML)
- Fast adaptation to new schemas / workloads  
- Improves generalization for unseen queries, logs, and SQL sequences  

### Robustness Testing
Includes:
- Gaussian noise  
- Feature masking  
- FGSM adversarial samples  
- Custom accuracy/F1 metrics  

### Latency Benchmarking
`bench/latency_test.py` measures:
- Mean latency  
- p90, p95, p99  
- GPU/CPU comparison  
- AMP support  

### Production Deployment
- Dockerfile for containerization  
- Kubernetes deployment, service, HPA  
- ConfigMap for central configuration  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/tll-rdbms.git
cd tll-rdbms

Training & Evaluation
Loop 1 — Supervised  Learning :   python scripts/train.py
Loop 2 — Reinforcement Learning : python scripts/train_rl.py
Loop 3 — Meta-Learning (MAML) :   python scripts/meta_adapt.py
Full Evaluation Pipeline      :   python scripts/eval.py
Testing & Quality Assurance - End-to-end test: python tests/test_pipeline.py
Robustness tests              :   python tests/robustness.py
Latency benchmarking          :   python bench/latency_test.py --device cuda
Docker Usage - Build image    :   docker build -t tll-rdbms -f docker/Dockerfile .
Run container                 :   docker run -p 8000:8000 tll-rdbms
Kubernetes Deployment         :   kubectl apply -f k8s/configmap.yaml
                              :   kubectl apply -f k8s/deployment.yaml
                              :   kubectl apply -f k8s/service.yaml
                              :   kubectl apply -f k8s/hpa.yaml

License

MIT License — freely available for academic, research, and commercial use.

Contributing

Contributions, issues, improvements, and suggestions are welcome!
Submit a PR or open an issue to join development.






