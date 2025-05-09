'''# Dynamic Community Detection Framework

Forks from https://github.com/aaamini/hsbm, which is the original code for "Hierarchical Stochastic Block Models for Multiplex Networks"

This repository provides a Python-based framework for experimenting with dynamic community detection algorithms on evolving networks. It includes implementations for network generation (e.g., Markovian evolution, Contracting Network Models), wrappers for inference algorithms (such as HBSM, DP-SBM, and PisCES, often leveraging R implementations via `rpy2`), and tools for evaluation and visualization of community structures.

The framework is designed to be extensible, allowing users to easily add new network generation models, integrate different inference techniques, and design custom experiments.

## Core Concepts

*   **`EvolvingCommunityModel` (`model.py`):** The central class that encapsulates the evolving network. It stores the generated adjacency matrices across layers, the true (ground truth) community assignments, and the results from various inference methods (learned community trajectories and MCMC sample histories).
*   **Network Generation:**
    *   **Markovian Evolution:** Nodes change communities based on a transition probability matrix (`generate_markov_network` in `model.py`).
    *   **Contracting Network Model (CNM):** Communities evolve based on a Brownian motion process, leading to contractions and expansions (`generate_contracting_network` in `model.py`).
*   **Inference Methods (`inference/`):**
    *   Wrappers call underlying algorithms (often R scripts) to infer community structures from the generated networks.
    *   Results include both a final 2D (nodes x layers) community assignment matrix and, where applicable, the full 3D (nodes x layers x iterations) history of MCMC samples or iterative assignments. These are stored in `model.learned_trajectories` and `model.mcmc_sample_history` respectively.
*   **Evaluation (`eval.py`):**
    *   Provides tools to assess the performance of inference methods against ground truth, primarily using Adjusted Mutual Information (AMI).
    *   The `Evaluation` class and associated functions allow for aggregate and slice-wise (per-layer) AMI calculations.
    *   The `if __name__ == "__main__":` block in `eval.py` runs a sample evaluation, sweeping through different numbers of communities (K) and plotting AMI results.
*   **Visualization (`visualize.py`):**
    *   Offers functions to plot true and learned community assignments.
    *   Supports interactive visualization of 3D MCMC/iterative histories with sliders to explore different iterations.
    *   The `if __name__ == "__main__":` block provides a demonstration of its capabilities.
           
      
