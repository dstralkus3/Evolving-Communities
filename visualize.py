from model import EvolvingCommunityModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider # Import the Slider widget

class Visualizer():

    def __init__(self, model_instance: EvolvingCommunityModel):
        self.model  = model_instance

        if self.model.dynamicNetwork is None:
            raise ValueError("Dynamic network is not set. Please generate a dynamic network first.")
        

    def plot_true_community_evolution(self):
        """Plots the ground truth community evolution over layers for each node."""
        plt.figure(figsize=(12, 6))
        # community_assignments shape is (N, T)
        N, T = self.model.community_assignments.shape
        for i in range(N):
            # Plot community of node i across all layers T
            plt.plot(range(T), self.model.community_assignments[i, :], marker='.', linestyle='-', label=f"Node {i}")
        
        plt.title('True Community Evolution Over Layers')
        plt.xlabel('Layer (Time Step)')
        plt.ylabel('Community ID')
        # Only show legend if few nodes, otherwise it gets cluttered
        if N <= 15:
             plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        else:
             plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def visualize_learned_community_evolution(self):
        """Plots learned community evolution interactively with an iteration slider."""
        if self.model.learned_trajectories is None:
            raise ValueError("Learned trajectories have not been set. Run inference first.")

        # learned_trajectories shape is assumed (N, T, Iterations)
        N, T, Iters = self.model.learned_trajectories.shape
        if Iters == 0:
            print("Learned trajectories have 0 iterations. Cannot plot.")
            return

        # --- Plot Setup ---
        # Create figure and main axes, leaving space at the bottom for the slider
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.2)

        # --- Initial Plot (Iteration 0) ---
        initial_assignments = self.model.learned_trajectories[:, :, 0] # Shape (N, T)
        lines = [] # Store line objects for updating
        for i in range(N):
            line, = ax.plot(range(T), initial_assignments[i, :], marker='.', linestyle='-', label=f"Node {i}")
            lines.append(line)

        ax.set_title(f'Learned Community Evolution (Iteration 0 / {Iters-1})')
        ax.set_xlabel('Layer (Time Step)')
        ax.set_ylabel('Community ID')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Determine y-axis limits based on all possible communities
        all_comms = np.unique(self.model.learned_trajectories)
        if len(all_comms) > 0:
            ax.set_ylim(min(all_comms)-0.5, max(all_comms)+0.5)

        if N <= 15:
             legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
             # Adjust layout calculation needs fig instead of plt?
             fig.tight_layout(rect=[0, 0, 0.85, 1])
        else:
             fig.tight_layout(rect=[0, 0.1, 1, 0.95]) # Leave space for slider

        # --- Slider Setup ---
        # Define axes for slider [left, bottom, width, height]
        ax_slider = fig.add_axes([0.15, 0.05, 0.65, 0.03])

        # Create the slider
        iteration_slider = Slider(
            ax=ax_slider,
            label='Iteration',
            valmin=0, # Start iteration
            valmax=Iters - 1, # End iteration
            valinit=0, # Initial iteration
            valstep=1, # Step by integer iterations
            valfmt='%d' # Format display value as integer
        )

        # --- Update Function --- 
        def update(val):
            iteration_index = int(iteration_slider.val)
            # Get data for the selected iteration
            assignments_at_iter = self.model.learned_trajectories[:, :, iteration_index]
            # Update the y-data for each line
            for node_idx, line in enumerate(lines):
                line.set_ydata(assignments_at_iter[node_idx, :])
            # Update the title
            ax.set_title(f'Learned Community Evolution (Iteration {iteration_index} / {Iters-1}) with {self.model.inference_method}' )
            # Redraw the figure
            fig.canvas.draw_idle()

        # Register the update function with the slider
        iteration_slider.on_changed(update)

        # Keep a reference to the slider (optional, prevents garbage collection in some environments)
        self._slider = iteration_slider

        plt.show()


if __name__ == "__main__":
    
    rng = np.random.default_rng(seed=40)

    ## PARAMETERS FOR EVOLVING COMMUNITY INSTANCE
    n_nodes = 20
    n_layers = 100
    num_communities = 100  # Changed from 10 to match transition matrix dimensions
    community_matrix = rng.normal(0, 1, (num_communities, num_communities))
    initial_distribution = rng.dirichlet(np.ones(num_communities), size = 1)[0]

    # MARKOV NETWORK HYPERPARAMETERS
    temperature = 100

    # Make transition matrix diagonal heavy for smoother paths
    transition_matrix = np.zeros((num_communities, num_communities))
    for i in range(num_communities):
        alpha = np.ones(num_communities)
        alpha[i] = temperature
        transition_matrix[i, :] = rng.dirichlet(alpha)

    # INITIALIZE MODEL AND PERFORM INFERENCE #
    model = EvolvingCommunityModel(n_nodes, n_layers, num_communities, community_matrix, initial_distribution)
    model.generate_markov_network(transition_matrix)

    # METHOD HYPERPARAMETERS #
    num_iters = 50
    model.learn_community_dynamics('dpsbm', num_iters = num_iters)
    
    # VISUALIZE RESULTS #
    visualizer = Visualizer(model)
    visualizer.plot_true_community_evolution()
    visualizer.visualize_learned_community_evolution()
