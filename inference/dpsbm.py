import numpy as np
import time
import sys
import os

# Add parent directory to path to import model
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- rpy2 setup ---
RPY2_LOADED = False
R_Matrix = None
R_hsbm = None
R_robjects = None
R_numpy2ri = None
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    R_robjects = robjects
    R_numpy2ri = numpy2ri
    R_Matrix = importr('Matrix')
    R_hsbm = importr('hsbm') # Assumes R package 'hsbm' is installed
    R_numpy2ri.activate()
    RPY2_LOADED = True
    print("rpy2 and R packages (Matrix, hsbm) loaded successfully for dpsbm.py.")
except ImportError as e:
    print(f"Error importing rpy2 or R packages for dpsbm.py: {e}")
    print("Please ensure rpy2 is installed (pip install rpy2) and the R packages 'Matrix' and 'hsbm' are installed in your R environment.")
except Exception as e: # Catch other R interface errors
    print(f"Error loading R packages (Matrix, hsbm) via rpy2: {e}")
# --- End rpy2 setup ---

def run_dpsbm_inference(model, **kwargs):
    """
    Runs DP-SBM inference using the R 'hsbm' package via rpy2.
    Populates model.learned_trajectories with assignment history (N, T, Iters).
    Populates model.learned_community_matrix with None (method doesn't estimate eta).

    Parameters:
    -----------
    model : EvolvingCommunityModel
        The model instance containing the dynamic network.
    **kwargs : dict
        Hyperparameters for hsbm::fit_mult_dpsbm (e.g., niter, Zcap, gam0, verb).
    """
    model.inference_method = 'dpsbm'

    if not RPY2_LOADED:
        print("Cannot run DP-SBM inference because rpy2 or R packages failed to load.")
        model.learned_trajectories = None
        model.learned_community_matrix = None
        return

    if model.dynamicNetwork is None:
        raise ValueError("Model dynamic network not generated. Call model.generate_markov_network() first.")

    print("\n--- Running DP-SBM Inference (via R) ---")
    start_time = time.time()

    # --- Prepare Input for R ---
    T = model.n_layers
    N = model.n_nodes
    K_true = model.num_communities # For setting Zcap if needed
    A_tensor = model.dynamicNetwork # Shape (N, N, T)
    A_list_np = [A_tensor[:, :, t].astype(np.float64) for t in range(T)] # Ensure float
    try:
        r_A_list = R_robjects.ListVector({
            str(i+1): R_Matrix.Matrix(A, sparse=True)
            for i, A in enumerate(A_list_np)
        })
    except Exception as e:
         print(f"Error converting numpy arrays to R sparse matrices: {e}")
         model.learned_trajectories = None
         model.learned_community_matrix = None
         return

    # --- Set Hyperparameters ---
    niter = int(kwargs.get('niter', 50))
    zcap = int(kwargs.get('Zcap', K_true + 5))
    gam0 = float(kwargs.get('gam0', 0.5))
    verb = bool(kwargs.get('verb', False))

    # --- Call R Function ---
    # Assuming the function is hsbm::fit_mult_dpsbm based on previous wrapper
    model.learned_trajectories = None # Reset
    model.learned_community_matrix = None # Reset (DP-SBM doesn't estimate eta)
    print(f"Calling hsbm::fit_mult_dpsbm with niter={niter}, Zcap={zcap}...")
    try:
        dpsbm_result_r = R_hsbm.fit_mult_dpsbm(
            r_A_list,
            gam0=R_robjects.FloatVector([gam0]),
            niter=R_robjects.IntVector([niter]),
            Zcap=R_robjects.IntVector([zcap]),
            verb=R_robjects.BoolVector([verb])
        )

        # --- Extract Trajectory (zb_list) ---
        print("Extracting results from R object...")
        # The structure might be slightly different, check R function's return value
        # Assuming it returns a list with zb_list directly accessible
        zb_list_r = dpsbm_result_r.rx2('zb_list')
        if zb_list_r != R_robjects.NULL:
            num_iters_r = len(zb_list_r)
            num_layers_r = len(zb_list_r[0])
            num_nodes_r = len(zb_list_r[0][0])
            print(f"  Raw zb_list dimensions (iters, layers, nodes): ({num_iters_r}, {num_layers_r}, {num_nodes_r})")

            trajectory_raw = np.zeros((num_iters_r, num_layers_r, num_nodes_r), dtype=int)
            for i in range(num_iters_r):
                for l in range(num_layers_r):
                    trajectory_raw[i, l, :] = np.array(zb_list_r[i][l], dtype=int) - 1 # 0-based

            model.learned_trajectories = trajectory_raw.transpose(2, 1, 0)
            print(f"  Stored DP-SBM trajectory with shape: {model.learned_trajectories.shape}")
        else:
            print("  Warning: DP-SBM R result did not contain 'zb_list'. Trajectories not stored.")

    except Exception as e:
        print(f"  Error during R hsbm::fit_mult_dpsbm execution or result extraction: {e}")
        import traceback
        traceback.print_exc()

    end_time = time.time()
    print(f"--- DP-SBM Inference finished in {end_time - start_time:.2f}s ---")
