from model import EvolvingCommunityModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # Import the Slider widget
from utils import simple_rw_transition, locality_sbm
import networkx as nx

class Visualizer():

    def __init__(self, model_instance: EvolvingCommunityModel):
        self.model = model_instance

        if self.model.dynamicNetwork is None:
            raise ValueError("Dynamic network is not set. Please generate a dynamic network first.")
        if self.model.true_community_assignments is None:
             raise ValueError("True community assignments are not set.")

    def plot_true_community_evolution(self, ax):
        """Plots the ground truth community evolution on a given Axes object."""
        # true_community_assignments shape is (N, T)
        N, T = self.model.true_community_assignments.shape
        for i in range(N):
            # Plot community of node i across all layers T
            ax.plot(range(T), self.model.true_community_assignments[i, :], marker='.', linestyle='-', label=f"Node {i}")

        ax.set_title('True Community Evolution')
        ax.set_xlabel('Layer (Time Step)')
        ax.set_ylabel('Community ID')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Only show legend if few nodes, otherwise it gets cluttered
        if N <= 15:
             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def visualize_learned_trajectories(self):
        """
        Plots true community evolution and learned trajectories for multiple methods.
        Subplots are arranged horizontally (left-to-right).
        Each method with 3D MCMC/iterative data gets its own iteration slider.
        """
        # true_community_assignments existence is guaranteed by __init__
        
        methods_to_plot = [] # Stores names of *learned* methods with valid data
        # Populate methods_to_plot, prioritizing 3D MCMC data then 2D MAP data
        if hasattr(self.model, 'mcmc_sample_history') and isinstance(self.model.mcmc_sample_history, dict):
            for key, val in self.model.mcmc_sample_history.items():
                if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[0] > 0 and val.shape[1] > 0 and val.shape[2] > 0: # N, T, Iters
                    methods_to_plot.append(key)
        
        if hasattr(self.model, 'learned_trajectories') and isinstance(self.model.learned_trajectories, dict):
            for key, val in self.model.learned_trajectories.items():
                if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] > 0 and val.shape[1] > 0: # N, T
                    if key not in methods_to_plot: # Add if not already from MCMC history (or if MCMC was invalid)
                        methods_to_plot.append(key)
        
        if not methods_to_plot and self.model.true_community_assignments is None:
            print("No true or learned trajectories/MCMC history available to visualize.")
            return
        if not methods_to_plot and self.model.true_community_assignments is not None:
            print("No learned trajectories or MCMC history available. Plotting only true communities.")
            # Fall through to plot only true if no learned methods are found but true exists.

        num_learned_methods = len(methods_to_plot)
        num_cols = 1 + num_learned_methods # 1 for true, others for learned

        fig, all_axes_flat = plt.subplots(1, num_cols, figsize=(8 * num_cols, 7), sharey=True) 
        
        if num_cols == 0: # Should only happen if true_community_assignments is also None
             plt.close(fig)
             return
        elif num_cols == 1: # Only true plot or one plot total
            all_axes = [all_axes_flat] 
        else:
            all_axes = all_axes_flat

        self._slider_refs = [] 

        # --- Calculate Global Y-axis Limits ---
        all_min_c_global, all_max_c_global = [np.inf], [-np.inf] 
        def update_global_yrange(data_array):
            if data_array is not None and data_array.size > 0:
                current_min = np.min(data_array)
                current_max = np.max(data_array)
                if current_min < all_min_c_global[0]: all_min_c_global[0] = current_min
                if current_max > all_max_c_global[0]: all_max_c_global[0] = current_max
        
        if self.model.true_community_assignments is not None:
            update_global_yrange(self.model.true_community_assignments)

        for method_name_for_yrange in methods_to_plot:
            data_to_check = None
            if hasattr(self.model, 'mcmc_sample_history') and method_name_for_yrange in self.model.mcmc_sample_history and \
               isinstance(self.model.mcmc_sample_history[method_name_for_yrange], np.ndarray) and \
               self.model.mcmc_sample_history[method_name_for_yrange].ndim == 3:
                data_to_check = self.model.mcmc_sample_history[method_name_for_yrange]
            elif hasattr(self.model, 'learned_trajectories') and method_name_for_yrange in self.model.learned_trajectories and \
                 isinstance(self.model.learned_trajectories[method_name_for_yrange], np.ndarray) and \
                 self.model.learned_trajectories[method_name_for_yrange].ndim == 2:
                 data_to_check = self.model.learned_trajectories[method_name_for_yrange]
            update_global_yrange(data_to_check)
        
        min_c_final = 0 if all_min_c_global[0] == np.inf else all_min_c_global[0]
        max_c_final = 0 if all_max_c_global[0] == -np.inf else all_max_c_global[0]
        
        # --- Plot Ground Truth (First Column if true_community_assignments exists) --- 
        ax_true = all_axes[0]
        N_ref, T_ref = 0,0 # Reference N, T for learned plots if true is missing
        
        if self.model.true_community_assignments is not None:
            N_ref, T_ref = self.model.true_community_assignments.shape
            self.plot_true_community_evolution(ax_true)
            ax_true.set_ylim(min_c_final - 0.5, max_c_final + 0.5 if max_c_final >= min_c_final else min_c_final + 1.5)
        else: # No true assignments, first plot will be the first learned method
            ax_true.set_title("True communities not available")
            ax_true.axis('off') # Hide this axis if no true data

        # --- Plot Each Learned Trajectory ---
        # If true_community_assignments is None, learned methods start at index 0 of all_axes
        # Otherwise, they start at index 1.
        learned_plot_offset = 0 if self.model.true_community_assignments is None else 1

        for i, method_name in enumerate(methods_to_plot):
            ax_idx = i + learned_plot_offset
            if ax_idx >= len(all_axes): # Safety check if only true was plotted and methods_to_plot was empty
                print(f"Warning: Attempting to plot method '{method_name}' but not enough axes available. Skipping.")
                continue
            ax_learned = all_axes[ax_idx]
            
            trajectory_data_to_plot = None
            is_3d_data = False
            num_iters_for_method = 0
            N_method, T_method = N_ref, T_ref 

            # Check for 3D MCMC/iterative data first
            if hasattr(self.model, 'mcmc_sample_history') and isinstance(self.model.mcmc_sample_history.get(method_name), np.ndarray) and \
               self.model.mcmc_sample_history[method_name].ndim == 3:
                mcmc_data = self.model.mcmc_sample_history[method_name]
                if mcmc_data.size > 0 and mcmc_data.shape[2] > 0:
                    trajectory_data_to_plot = mcmc_data
                    N_method, T_method, num_iters_for_method = trajectory_data_to_plot.shape
                    is_3d_data = True
                    print(f"Visualizing 3D data for method '{method_name}' (N={N_method}, T={T_method}, Iters={num_iters_for_method}).")
                else:
                    print(f"MCMC/iterative history for '{method_name}' is empty or invalid. Falling back.")
            
            # Fallback to 2D MAP data
            if not is_3d_data:
                if hasattr(self.model, 'learned_trajectories') and isinstance(self.model.learned_trajectories.get(method_name), np.ndarray) and \
                   self.model.learned_trajectories[method_name].ndim == 2:
                    map_data = self.model.learned_trajectories[method_name]
                    if map_data.size > 0:
                        trajectory_data_to_plot = map_data
                        N_method, T_method = trajectory_data_to_plot.shape
                        num_iters_for_method = 0 # No iterations for 2D data
                        is_3d_data = False # Explicitly false
                        print(f"Visualizing 2D MAP trajectory for method '{method_name}' (N={N_method}, T={T_method}).")
                    else:
                         print(f"2D MAP trajectory for '{method_name}' is empty.")
                else:
                    if not is_3d_data: # only print if no 3D data was found
                        print(f"No suitable 2D data found for method '{method_name}'.")

            if trajectory_data_to_plot is None:
                print(f"No valid data found for method '{method_name}'. Skipping plot for this method.")
                ax_learned.set_title(f'No data for {method_name}')
                ax_learned.set_xlabel('Layer (Time Step)')
                ax_learned.set_ylabel('')
                ax_learned.axis('off')
                continue
            
            if N_method == 0 or T_method == 0:
                print(f"Error: Node count (N={N_method}) or Layer count (T={T_method}) is zero for method '{method_name}'. Skipping plot.")
                ax_learned.set_title(f'Invalid data dims for {method_name}')
                ax_learned.set_xlabel('Layer (Time Step)')
                ax_learned.set_ylabel('')
                ax_learned.axis('off')
                continue

            lines_for_method = []
            initial_assignments = trajectory_data_to_plot[:, :, 0] if is_3d_data and num_iters_for_method > 0 else trajectory_data_to_plot
            
            if initial_assignments.shape[0] != N_method or initial_assignments.shape[1] != T_method:
                 print(f"Error: Shape mismatch for initial assignments for {method_name}. Expected ({N_method},{T_method}), got {initial_assignments.shape}. Skipping.")
                 ax_learned.set_title(f'Data shape error for {method_name}')
                 ax_learned.axis('off')
                 continue

            for node_idx in range(N_method):
                line, = ax_learned.plot(range(T_method), initial_assignments[node_idx, :], marker='.', linestyle='-', label=f"Node {node_idx}")
                lines_for_method.append(line)
            
            base_title = f'Learned: {method_name}'
            current_title = f'{base_title} - Iter 0 / {max(0, num_iters_for_method-1)}' if is_3d_data and num_iters_for_method > 0 else base_title
            ax_learned.set_title(current_title)
            ax_learned.set_xlabel('Layer (Time Step)')
            ax_learned.set_ylabel('') 
            ax_learned.grid(axis='y', linestyle='--', alpha=0.7)
            if N_method <= 15:
                ax_learned.legend(loc='center left', bbox_to_anchor=(1.02, 0.5)) # Adjust legend position slightly
            
            # Ensure Y-limits are consistent if this plot is the first one (i.e., no true plot)
            if learned_plot_offset == 0 and i == 0:
                 ax_learned.set_ylim(min_c_final - 0.5, max_c_final + 0.5 if max_c_final >= min_c_final else min_c_final + 1.5)


            if is_3d_data and num_iters_for_method > 1:
                bbox = ax_learned.get_position()
                slider_ax_rect = [bbox.x0 + bbox.width * 0.1, bbox.y0 - 0.12, bbox.width * 0.8, 0.03] # Adjusted y0 for more space
                
                ax_slider_method = fig.add_axes(slider_ax_rect)
                iteration_slider_method = Slider(
                    ax=ax_slider_method,
                    label=f'{method_name[:10]} Iter', # Shorten label if too long
                    valmin=0,
                    valmax=num_iters_for_method - 1,
                    valinit=0,
                    valstep=1,
                    valfmt='%d'
                )

                def make_update_func(data, lines, ax, title_base, total_iters_func): # Pass total_iters as a function
                    def update(val):
                        iter_idx = int(val) 
                        assignments = data[:, :, iter_idx]
                        for node_idx_func, line_obj_func in enumerate(lines):
                            line_obj_func.set_ydata(assignments[node_idx_func, :])
                        ax.set_title(f'{title_base} - Iter {iter_idx} / {max(0, total_iters_func-1)}')
                        fig.canvas.draw_idle()
                    return update
                
                update_func = make_update_func(trajectory_data_to_plot, lines_for_method, ax_learned, base_title, num_iters_for_method)
                iteration_slider_method.on_changed(update_func)
                self._slider_refs.append(iteration_slider_method) 

        fig.tight_layout(rect=[0, 0.08 if self._slider_refs else 0.03, 0.95, 0.95]) # Adjust bottom for sliders
        plt.show()

    def plot_community_matrix(self, learned: bool = False, ax=None):
        """
        Displays the community matrix as a heatmap. If learned=True and a learned matrix
        is available, it plots that; otherwise it plots the true community_matrix.
        """
        # Choose which matrix to display
        if learned:
            matrix = self.model.learned_community_matrix
            title = 'Learned Community Matrix'
            if matrix is None:
                print("No learned community matrix available to display.")
                return
        else:
            matrix = self.model.community_matrix
            title = 'True Community Matrix'

        # Create new figure and axes if none provided
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Community')
        ax.set_ylabel('Community')
        plt.colorbar(im, ax=ax)

        if created_fig:
            plt.show()

    def display_temporal_network(self):
        """
        Displays the network over time using Matplotlib + NetworkX and a Slider.
        Nodes are colored by their true community assignments.
        """
        # Extract adjacency tensor and community assignments
        A = self.model.dynamicNetwork  # shape (N, N, T)
        com = self.model.true_community_assignments  # shape (N, T)
        N, _, T = A.shape

        # Consistent circular layout
        agg_graph = nx.from_numpy_array(A.sum(axis=2))
        pos = nx.circular_layout(agg_graph)

        # Prepare figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplots_adjust(bottom=0.2)

        def draw_snapshot(t):
            ax.clear()
            Gt = nx.from_numpy_array(A[:, :, t])
            # Draw edges
            nx.draw_networkx_edges(Gt, pos, ax=ax, edge_color='gray', alpha=0.5)
            # Draw nodes with community color
            colors = [com[i, t] for i in range(N)]
            nx.draw_networkx_nodes(
                Gt, pos, ax=ax,
                node_color=colors, cmap='viridis',
                vmin=com.min(), vmax=com.max(),
                node_size=100
            )
            ax.set_title(f'Temporal Network (t={t})')
            ax.axis('off')

        # Initial draw
        draw_snapshot(0)

        # Slider for time
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Time', 0, T-1, valinit=0, valstep=1)

        def update(val):
            draw_snapshot(int(val))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

if __name__ == "__main__":

    seed = 42
    rng = np.random.default_rng(seed=seed)

    # --- Parameters ---
    n_nodes = 30
    n_layers = 20
    num_communities = 40

    # GENERATE COMMUNITY MATRIX
    community_matrix = locality_sbm(num_communities, beta =.05)

    transition_matrix = simple_rw_transition(num_communities)
    initial_distribution = np.ones(num_communities) / num_communities 
 
    model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)
    model.generate_contracting_network()
 
    ## Learn Model
    model.learn_community_dynamics(method = 'hbsm', niter = 100)
 
    ## Visualize
    visualizer = Visualizer(model)
    visualizer.visualize_learned_trajectories()
    visualizer.display_temporal_network()