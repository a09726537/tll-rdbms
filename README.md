# TLL-RDBMS: Triple Loop Learning for Intelligent Anomaly Detection in Databases

This repository contains the code, poster, and dataset setup for the paper:

**"Intelligent Anomaly Detection in Database Security: A Triple Loop Learning Framework"**  
Presented at ICSC 2025 by William K. (University of Vienna)

## Overview

The TLL framework combines:
- **Operational loop:** SQL anomaly detection using Kitsune-style autoencoders
- **Tactical loop:** DQN-based reinforcement learning for adaptive thresholds
- **Strategic loop:** Few-shot generalization using MAML meta-learning

## Files

- `poster/Tll_Icsc2025_Poster.tex`: LaTeX source of the conference poster
- `images/`: Logos and experiment result plots
- `datasets/`: Placeholder for UNSW-NB15, SQL logs, and MAD-GAN data
- `notebooks/`: Jupyter/PyTorch code examples (to be added)

## Setup

```bash
git clone https://github.com/a09726537/tll-rdbms.git
cd tll-rdbms
pip install -r requirements.txt
```

## Contact

Author: William K.  
Email: a09726537@unet.univie.ac.at  
GitHub: https://github.com/a09726537/tll-rdbms

## License

For academic use only. Licensed under an open-source academic license.
