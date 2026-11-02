import os
import numpy as np
import matplotlib.pyplot as plt
from slope_entropy import compute_slope_entropy

DATA_FILE = os.path.join("processed_data", "bearing1_hybrid_dataset.npz")
LATEX_DIR = "latex"

def main():
    print("=== PARAMETER SENSITIVITY ANALYSIS ===")
    
    if not os.path.exists(DATA_FILE):
        print("Dataset not found.")
        return
        
    dataset = np.load(DATA_FILE)
    data = dataset['data']
    n_records = data.shape[0]
    
    # We will test the sensitivity of the algorithm to the Gamma threshold
    # Default is 1.0. We will test 0.5, 1.0, and 1.5
    gamma_values = [0.5, 1.0, 1.5]
    window_size = 2048
    n_windows = data.shape[1] // window_size
    
    results = {}
    
    for gamma in gamma_values:
        print(f"Computing evolution for gamma = {gamma}...")
        evolution = []
        for i in range(n_records):
            record_data = data[i]
            window_entropies = []
            for w in range(n_windows):
                window = record_data[w*window_size : (w+1)*window_size]
                e = compute_slope_entropy(window, m=3, gamma=gamma, delta=0.1)
                window_entropies.append(e)
            evolution.append(np.mean(window_entropies))
        results[gamma] = evolution

    # Plotting the sensitivity analysis
    print("Generating sensitivity plot...")
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(n_records)
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    markers = ['^', 'o', 's']
    
    for idx, gamma in enumerate(gamma_values):
        plt.plot(x_axis, results[gamma], marker=markers[idx], linestyle='-', 
                 color=colors[idx], markersize=4, label=f'$\gamma = {gamma}$')
                 
    # Mark the known turning point (index 42)
    plt.axvline(x=42, color='red', linestyle='--', linewidth=2, label='Known Turning Point')
    
    plt.title("Parameter Sensitivity Analysis (Varying $\gamma$)", fontsize=14, fontweight='bold')
    plt.xlabel("Record Index", fontsize=12)
    plt.ylabel("Slope Entropy", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    
    fig_path = os.path.join(LATEX_DIR, "parameter_sensitivity.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Saved plot to {fig_path}")

if __name__ == "__main__":
    main()
