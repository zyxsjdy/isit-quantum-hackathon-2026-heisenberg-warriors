# Quantum Optimization for UAV Swarm Deployment in ISAC-Assisted Disaster Relief

**Team:** Heisenberg Warriors  
**Authors:** Tinh T. Bui, Quan T. Dao, Sasinda C. Prabhashana, Yuxiang Zheng, Nhien Q. T. Thoong, Erick Oduniyi

This repository contains a QAOA-based simulation for optimizing UAV swarm deployment in an integrated sensing and communication (ISAC) assisted disaster-relief scenario. The main notebook formulates UAV placement as a QUBO problem, converts it to an Ising Hamiltonian, solves it with QAOA, and compares the result against brute-force, greedy, and random baselines.

## Main File

- `QAOA_ISAC_UAV 2.ipynb` - main implementation and experiment notebook.

## What the Notebook Does

1. Builds the ISAC disaster-relief scenario with UAVs, candidate grid points, survivors, channel gains, MRT beamforming, and SINR/rate calculations.
2. Constructs the QUBO objective with deployment constraints.
3. Converts the QUBO to a Qiskit `QuadraticProgram` and Ising operator.
4. Runs a brute-force classical baseline for validation.
5. Builds and optimizes a QAOA circuit using COBYLA.
6. Samples the optimized circuit and decodes the best UAV placement.
7. Compares QAOA with brute-force, greedy, and random baselines using rate and approximation ratio.

## Requirements

Install the main Python packages before running the notebook:

```bash
pip install numpy scipy matplotlib qiskit qiskit-aer qiskit-optimization qiskit-ibm-runtime
```

## How to Run

Open `QAOA_ISAC_UAV 2.ipynb` in Jupyter Notebook or JupyterLab and run the cells from top to bottom.

By default, use simulator mode. To run on IBM Quantum hardware, edit the backend-selection cell, set `BACKEND_MODE = 'hardware'`

## Outputs

The notebook produces convergence plots, decoded UAV placement results, comparison tables, and bar charts. Existing result figures and PDFs in the folder are generated artifacts from the experiments.
