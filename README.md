'''# Dynamic Community Detection Framework

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
*   **Figure 3a Replication (`fig3a.py`):**
    *   A dedicated script to replicate Figure 3a from the Hierarchical Bayesian Stochastic Blockmodel (HBSM) paper, plotting AMI vs. a network evolution parameter `tau`.

## Repository Structure

```
.
├── README.md                   # This file
├── model.py                    # Core EvolvingCommunityModel class
├── fig3a.py                    # Script for HBSM Figure 3a replication
├── eval.py                     # Evaluation script (e.g., AMI vs. K)
├── visualize.py                # Visualization tools for community trajectories
├── utils.py                    # Utility functions (e.g., SBM generation)
├── requirements.txt            # Python dependencies
├── inference/                  # Wrappers for inference algorithms
│   ├── hiersbm.py              # HBSM wrapper (uses R)
│   ├── dpsbm.py                # DP-SBM wrapper (uses R)
│   ├── pisces.py               # PisCES wrapper (uses R + Python K-Means)
│   └── __init__.py
└── hbsm/                       # R project for HBSM and related methods
    ├── hsbm_package/           # R package for hsbm
    │   ├── R/                  # R source files (including test_models.R, competing_methods.R)
    │   ├── man/
    │   ├── DESCRIPTION
    │   └── NAMESPACE
    └── markov_experim.R        # Original R script for Markov experiments (reference)
```

## Setup and Installation

### 1. Python Environment

It's recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. R Setup

This project relies on R and several R packages, primarily accessed via the `rpy2` Python library.

*   **Install R:** Download and install R from [CRAN](https://cran.r-project.org/).
*   **R Packages:** Open an R console and install the following packages:
    ```R
    install.packages(c("Matrix", "igraph", "irlba", "Rcpp", "RcppArmadillo", "devtools", "nett"))
    ```
*   **Custom `hsbm` R package:** Install the local `hsbm` package provided in this repository:
    ```R
    # In an R console, navigate to the root of this project if devtools requires it
    # or provide the full path to the hsbm_package directory.
    # Example:
    # setwd("/path/to/your/project/code") 
    # devtools::install_local("hbsm/hsbm_package") 
    # Or directly:
    devtools::install_local("/path/to/your/project/code/hbsm/hsbm_package") 
    ```
    Replace `/path/to/your/project/code/` with the actual absolute path to this repository on your system.

*   **`rpy2` Configuration:** Ensure `rpy2` can find your R installation. This is usually automatic if R is in your system's PATH. If you encounter issues, you might need to set the `R_HOME` environment variable.

## Running Existing Scripts

### Replicating Figure 3a (HBSM Paper)

This script runs the HBSM, DP-SBM, and PisCES inference methods on networks generated with varying `tau` values (controlling temporal smoothness) and plots the aggregate AMI for each method.

```bash
python3 fig3a.py
```
Output:
*   A plot saved as `fig3a_anmi_vs_tau.png`.
*   Console output detailing the process.
*   A visualization of true and learned communities for a specific `tau` (e.g., `tau=0.5`).

### Running General Model Evaluation

This script evaluates inference methods across a range of community counts (K), generating networks and plotting aggregate AMI.

```bash
python3 eval.py
```
Output:
*   A plot saved as `eval_k_vs_ami.png`.
*   Console output with warnings or progress.

### Running Visualizations

The `visualize.py` script has a self-contained example in its `__main__` block. It generates a contracting network, runs inference, and then displays the true and learned trajectories.

```bash
python3 visualize.py
```
Output:
*   A multi-panel plot showing true communities and learned communities from HBSM, DP-SBM, and PisCES, with interactive sliders for MCMC/iteration history if available.

## Extending the Codebase

This framework is designed for extension. Here's how you can add new components:

### 1. Adding a New Network Generation Model

1.  **Location:** Implement your new generation logic as a method within the `EvolvingCommunityModel` class in `model.py`.
2.  **Functionality:**
    *   The method should populate:
        *   `self.adj_matrices`: A list of `scipy.sparse.csr_matrix` objects, where each matrix represents the adjacency matrix of the network at a specific layer. Dimensions should be (N x N).
        *   `self.true_community_assignments`: A 2D NumPy array of shape (N x T), where N is the number of nodes and T is the number of layers. Each entry `(i, t)` should be an integer representing the true community assignment of node `i` at layer `t` (0-indexed).
    *   It can take parameters specific to your model (e.g., number of nodes, layers, communities, model-specific hyperparameters).
3.  **Example Signature:**
    ```python
    # In model.py, within EvolvingCommunityModel class
    def generate_my_new_network(self, param1, param2, seed=None):
        # ... your logic to generate adj_matrices and true_community_assignments ...
        self.adj_matrices = [...]
        self.true_community_assignments = np.array(...)
        # Optionally store other ground truth parameters
        # self.my_model_true_params = {...}
    ```

### 2. Adding a New Inference Method

1.  **Create a Wrapper Script:**
    *   Add a new Python file in the `inference/` directory (e.g., `my_method.py`).
    *   This script should define a primary function, for example:
        ```python
        # inference/my_method.py
        import numpy as np
        # Add any necessary imports for your method (e.g., rpy2, custom libraries)

        def run_my_method_inference(model_object, adj_matrices_list, num_communities_true, **kwargs):
            """
            Runs MyNewMethod inference.
            Args:
                model_object: The EvolvingCommunityModel instance.
                adj_matrices_list: List of SciPy sparse adjacency matrices.
                num_communities_true: The true number of communities (can be used as K_init).
                **kwargs: Additional method-specific hyperparameters.
            """
            print(f"--- Running MyNewMethod Inference with kwargs: {kwargs} ---")
            # Your inference logic here. This might involve:
            # - Calling an R script via rpy2
            # - Calling a C++ binary
            # - Implementing the algorithm directly in Python

            # Example placeholder results
            num_nodes = adj_matrices_list[0].shape[0]
            num_layers = len(adj_matrices_list)
            
            # Final 2D learned trajectory (nodes x layers)
            learned_traj_2d = np.random.randint(0, num_communities_true, size=(num_nodes, num_layers))
            
            # Full 3D MCMC/iterative history (nodes x layers x iterations), if applicable
            # For example, 50 iterations
            mcmc_history_3d = np.random.randint(0, num_communities_true, size=(num_nodes, num_layers, 50))

            # Store results in the model object
            model_object.learned_trajectories['my_method'] = learned_traj_2d
            model_object.mcmc_sample_history['my_method'] = mcmc_history_3d
            
            # Optionally store other learned parameters
            # model_object.learned_parameters['my_method'] = {'param_x': value_x}

            print("--- MyNewMethod Inference finished ---")
            return learned_traj_2d # Or some other primary result if needed by learn_community_dynamics
        ```

2.  **Integrate with `EvolvingCommunityModel`:**
    *   In `model.py`, import your new inference function.
    *   Modify the `learn_community_dynamics` method in `EvolvingCommunityModel` to call your new function when `method='my_method'` is passed.
    ```python
    # In model.py
    # ... other imports ...
    from inference.my_method import run_my_method_inference # Add this

    class EvolvingCommunityModel:
        # ...
        def __init__(self, ...):
            # ...
            self.learned_trajectories = {} # Ensure these are initialized
            self.mcmc_sample_history = {}
            self.learned_community_matrix = None # Or a dict if multiple methods learn it
            self.learned_parameters = {} # For other learned params
            self.inference_method = None
            # ...

        def learn_community_dynamics(self, method, **kwargs):
            # ...
            # existing methods (hbsm, dpsbm, pisces)
            # ...
            elif method == 'my_method':
                print(f"  learn_community_dynamics called for method '{method}' with kwargs: {kwargs}")
                # Ensure adj_matrices are generated
                if not self.adj_matrices:
                    raise ValueError("Adjacency matrices not generated. Call a generation method first.")
                
                # Pass relevant parts of self and kwargs to your inference function
                run_my_method_inference(
                    self, # Pass the model instance itself
                    self.adj_matrices,
                    self.num_communities, # True K
                    **kwargs
                )
                self.inference_method = method # Set the primary inference method run
            else:
                raise ValueError(f"Unknown inference method: {method}")

            if method not in self.learned_trajectories:
                 print(f"Warning: Learned trajectory for method '{method}' was not populated in self.learned_trajectories by its inference function.")
            if method not in self.mcmc_sample_history:
                 print(f"Warning: MCMC sample history for method '{method}' was not populated in self.mcmc_sample_history by its inference function.")

            return self # Allow chaining
    ```

### 3. Adding a New Experiment

1.  Create a new Python script (e.g., `my_new_experiment.py`) in the root directory.
2.  In this script:
    *   Import `EvolvingCommunityModel` from `model.py`.
    *   Import any necessary generation or utility functions.
    *   Instantiate `EvolvingCommunityModel` with desired parameters.
    *   Call a network generation method (e.g., `model.generate_markov_network(...)` or your custom one).
    *   Call `model.learn_community_dynamics(method='your_method_name', **method_kwargs)` for one or more inference methods.
    *   Use functions from `eval.py` (e.g., `Evaluation` class or standalone functions if refactored) to calculate performance metrics.
    *   Use functions from `visualize.py` (e.g., `visualize_learned_trajectories`) to plot results.
    *   Save plots or print summaries as needed.

### 4. Modifying Evaluation

*   The `eval.py` script uses `sklearn.metrics.adjusted_mutual_info_score`. You can add other metrics by:
    *   Modifying the `Evaluation` class methods (`aggregate_nmi`, `slice_nmi`) to compute and store additional scores.
    *   Alternatively, if `eval.py` is refactored into standalone functions, you can add new evaluation functions and call them in your experiment scripts.
    *   Ensure your new metrics handle the `model.true_community_assignments` and the 2D arrays from `model.learned_trajectories[method_name]`.

## Citing

If you use this framework in research that builds upon the Hierarchical Bayesian Stochastic Blockmodel, please consider citing the original HBSM paper:
*   *(Details of the original HBSM paper, e.g., Authors, Title, Journal, Year, DOI - Please add this if applicable)*

## Troubleshooting

*   **`rpy2` issues:**
    *   "R_HOME not found": Ensure R is installed and in your PATH, or set the `R_HOME` environment variable to your R installation directory.
    *   "Cannot find -lR": This can be an issue with R's shared library. Ensure R was compiled with shared library support. Consult `rpy2` documentation for OS-specific fixes.
    *   Package not found by `rpy2` but installed in R: Sometimes, R might have multiple library paths. Ensure `rpy2` is picking up the correct one or that packages are installed in a globally accessible R library.
*   **`devtools::install_local` fails:**
    *   Ensure you have R build tools installed (Rtools on Windows, Xcode command-line tools on macOS, `r-base-dev` or similar on Linux).
    *   Check for error messages from `devtools` regarding missing dependencies for the `hsbm` package itself.

## Contributing

Contributions are welcome! Please feel free to:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Submit a pull request with a clear description of your contributions.

---

This README provides a starting point. Feel free to expand it with more specific details as the project evolves!
