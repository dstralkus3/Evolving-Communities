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
        if self.model.community_assignments is None:
             raise ValueError("True community assignments are not set.")

    def plot_true_community_evolution(self, ax):
        """Plots the ground truth community evolution on a given Axes object."""
        # community_assignments shape is (N, T)
        N, T = self.model.community_assignments.shape
        for i in range(N):
            # Plot community of node i across all layers T
            ax.plot(range(T), self.model.community_assignments[i, :], marker='.', linestyle='-', label=f"Node {i}")

        ax.set_title('True Community Evolution')
        ax.set_xlabel('Layer (Time Step)')
        ax.set_ylabel('Community ID')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Only show legend if few nodes, otherwise it gets cluttered
        if N <= 15:
             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def visualize_learned_trajectories(self):
        """Plots true and learned community evolution side-by-side with an iteration slider."""
        if self.model.learned_trajectories is None:
            print("Learned trajectories have not been set. Cannot create comparison plot.")
            return

        # learned_trajectories shape is assumed (N, T, Iterations)
        N, T, Iters = self.model.learned_trajectories.shape
        if Iters == 0:
            print("Learned trajectories have 0 iterations. Cannot plot comparison.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        plt.subplots_adjust(bottom=0.2) # Make space for slider

        ax_true = axes[0]
        ax_learned = axes[1]

        # --- Plot Ground Truth (Left) --- 
        self.plot_true_community_evolution(ax_true)

        # --- Plot Initial Learned State (Right) --- 
        initial_assignments = self.model.learned_trajectories[:, :, 0] # Shape (N, T)
        lines_learned = [] # Store line objects for updating learned plot
        for i in range(N):
            line, = ax_learned.plot(range(T), initial_assignments[i, :], marker='.', linestyle='-', label=f"Node {i}")
            lines_learned.append(line)

        ax_learned.set_title(f'Learned Evolution ({self.model.inference_method or "Unknown"}, Iter 0 / {Iters-1})')
        ax_learned.set_xlabel('Layer (Time Step)')
        # ax_learned.set_ylabel('Community ID') # Y label is shared
        ax_learned.grid(axis='y', linestyle='--', alpha=0.7)
        # Don't show legend for learned plot if true plot has it or N is large
        if N <= 15:
             ax_learned.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Determine shared y-axis limits based on all possible communities
        all_comms_true = np.unique(self.model.community_assignments)
        all_comms_learned = np.unique(self.model.learned_trajectories)
        min_comm = min(np.min(all_comms_true) if len(all_comms_true)>0 else 0, 
                       np.min(all_comms_learned) if len(all_comms_learned)>0 else 0)
        max_comm = max(np.max(all_comms_true) if len(all_comms_true)>0 else 0, 
                       np.max(all_comms_learned) if len(all_comms_learned)>0 else 0)
        ax_true.set_ylim(min_comm - 0.5, max_comm + 0.5)
        # ax_learned uses the same ylim due to sharey=True

        # Adjust layout to prevent overlap, considering potential legends
        if N <= 15:
            # If legends are shown, need more space between plots and for right legend
             fig.tight_layout(rect=[0, 0.1, 0.9, 0.95], pad=2.0)
        else:
             fig.tight_layout(rect=[0, 0.1, 1, 0.95]) # Rect leaves space for slider

        # --- Slider Setup --- 
        ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03]) # Centered below plots
        iteration_slider = Slider(
            ax=ax_slider,
            label='Iteration',
            valmin=0,
            valmax=Iters - 1,
            valinit=0,
            valstep=1,
            valfmt='%d'
        )

        # --- Update Function --- 
        def update(val):
            iteration_index = int(iteration_slider.val)
            assignments_at_iter = self.model.learned_trajectories[:, :, iteration_index]
            # Update the y-data for each line in the learned plot
            for node_idx, line in enumerate(lines_learned):
                line.set_ydata(assignments_at_iter[node_idx, :])
            # Update the title of the learned plot
            ax_learned.set_title(f'Learned Evolution ({self.model.inference_method or "Unknown"}, Iter {iteration_index} / {Iters-1})')
            fig.canvas.draw_idle()

        iteration_slider.on_changed(update)
        self._slider = iteration_slider # Keep reference

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
        com = self.model.community_assignments  # shape (N, T)
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
    model.generate_markov_network(transition_matrix, seed = seed)
 
    ## Learn Model
    model.learn_community_dynamics(method = 'hbsm', niter = 100)
 
    ## Visualize
    visualizer = Visualizer(model)
    visualizer.visualize_learned_trajectories()
    