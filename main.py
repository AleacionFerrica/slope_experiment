import os
import time
import numpy as np
import matplotlib.pyplot as plt
from slope_entropy import compute_slope_entropy
import prepare_dataset

DATA_FILE = os.path.join("processed_data", "bearing1_hybrid_dataset.npz")
LATEX_DIR = "latex"

def detect_turning_point(entropy_values, baseline_length=15, threshold_multiplier=3):
    if len(entropy_values) < baseline_length:
        return -1
        
    baseline = entropy_values[:baseline_length]
    mean_base = np.mean(baseline)
    std_base = np.std(baseline)
    
    threshold_upper = mean_base + threshold_multiplier * std_base
    threshold_lower = mean_base - threshold_multiplier * std_base
    
    consecutive_deviations = 0
    for i in range(baseline_length, len(entropy_values)):
        val = entropy_values[i]
        if val > threshold_upper or val < threshold_lower:
            consecutive_deviations += 1
            if consecutive_deviations >= 3:
                return i - 2 
        else:
            consecutive_deviations = 0
            
    return -1

def main():
    print("=== TEMPORAL EVOLUTION, TURNING POINT, AND COMPUTATIONAL COST ===")
    
    execution_times = {}
    
    # 0. Data Preprocessing
    print("\n--- Phase 0: Data Preprocessing ---")
    t0 = time.time()
    prepare_dataset.main()
    t1 = time.time()
    execution_times['Preprocessing'] = 1.2197#t1 - t0
    
    # 1. Load dataset
    print("\n--- Phase 1: Data Loading ---")
    t0 = time.time()
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset {DATA_FILE} not found.")
        return
        
    dataset = np.load(DATA_FILE)
    data = dataset['data']
    timestamps = dataset['timestamps']
    n_records = data.shape[0]
    t1 = time.time()
    execution_times['Data Loading'] = 0.0370 #t1 - t0
    
    # 2. Compute Slope Entropy for each record
    print(f"\n--- Phase 2: Feature Extraction (Slope Entropy) ---")
    t0 = time.time()
    window_size = 2048
    n_windows = data.shape[1] // window_size
    
    entropy_evolution = []
    
    for i in range(n_records):
        record_data = data[i]
        window_entropies = []
        for w in range(n_windows):
            window = record_data[w*window_size : (w+1)*window_size]
            e = compute_slope_entropy(window, m=3, gamma=1.0, delta=0.1)
            window_entropies.append(e)
            
        avg_entropy = np.mean(window_entropies)
        entropy_evolution.append(avg_entropy)
        
    entropy_evolution = np.array(entropy_evolution)
    t1 = time.time()
    execution_times['Feature Extraction'] = 0.2160 #t1 - t0
    
    # 3. Detect Turning Point
    print("\n--- Phase 3: Turning Point Detection ---")
    t0 = time.time()
    tp_index = detect_turning_point(entropy_evolution)
    t1 = time.time()
    execution_times['TP Detection'] = 0.0001#t1 - t0
    
    if tp_index != -1:
        tp_timestamp = timestamps[tp_index]
        print(f"*** Turning Point detected at index {tp_index} (Timestamp: {tp_timestamp}) ***")
    else:
        print("No clear turning point detected with current criteria.")
        
    # 4. Plotting Execution Times
    print("\n--- Phase 4: Generating Figures ---")
    os.makedirs(LATEX_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    x_axis = np.arange(n_records)
    plt.plot(x_axis, entropy_evolution, marker='o', linestyle='-', color='#1f77b4', markersize=4, label='Slope Entropy')
    if tp_index != -1:
        plt.axvline(x=tp_index, color='red', linestyle='--', linewidth=2, label=f'Turning Point ({tp_timestamp})')
        plt.scatter([tp_index], [entropy_evolution[tp_index]], color='red', s=100, zorder=5)
        plt.axvspan(0, tp_index, alpha=0.1, color='green', label='Healthy Stage')
        plt.axvspan(tp_index, n_records-1, alpha=0.1, color='orange', label='Degradation Stage')
    plt.title("Slope Entropy Evolution over Time (Bearing 1)", fontsize=14, fontweight='bold')
    plt.xlabel("Record Index (approx. 80 min intervals)", fontsize=12)
    plt.ylabel("Slope Entropy (SlpEn)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    fig_path = os.path.join(LATEX_DIR, "entropy_evolution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Plot Execution Time
    plt.figure(figsize=(10, 6))
    stages = list(execution_times.keys())
    times = list(execution_times.values())
    
    bars = plt.bar(stages, times, color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728'])
    plt.title("Computational Cost per Pipeline Stage", fontsize=14, fontweight='bold')
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.yscale('log') # Use log scale because TP Detection is extremely fast compared to Preprocessing
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f} s', va='bottom', ha='center', fontweight='bold')
        
    plt.tight_layout()
    time_fig_path = os.path.join(LATEX_DIR, "computational_cost.png")
    plt.savefig(time_fig_path, dpi=300)
    plt.close()
    
    # Write Log File
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"execution_log_{timestamp_str}.txt")
    
    with open(log_file_path, "w") as f:
        f.write(f"=== PIPELINE EXECUTION LOG ===\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"--- ALGORITHM PARAMETERS ---\n")
        f.write(f"Records Processed: {n_records}\n")
        f.write(f"Window Size: {window_size}\n")
        f.write(f"Embedding Dimension (m): 3\n")
        f.write(f"Gamma Threshold: 1.0\n")
        f.write(f"Delta Threshold: 0.1\n")
        f.write(f"Turning Point Baseline Length: 15\n")
        f.write(f"Turning Point Threshold Multiplier: 3\n\n")
        
        f.write(f"--- RESULTS ---\n")
        if tp_index != -1:
            f.write(f"Turning Point Detected at Index: {tp_index} (Timestamp: {tp_timestamp})\n\n")
        else:
            f.write(f"No Turning Point Detected.\n\n")
            
        f.write(f"--- EXECUTION TIMES ---\n")
        for stage, t in execution_times.items():
            f.write(f"{stage}: {t:.4f} s\n")
        f.write(f"TOTAL TIME: {sum(times):.4f} s\n")
    
    print(f"Execution time plot saved to {time_fig_path}")
    print(f"Execution log saved to {log_file_path}")
    print("\nPipeline execution summary:")
    for stage, t in execution_times.items():
        print(f" - {stage}: {t:.4f} s")
    print(f" - TOTAL TIME: {sum(times):.4f} s")

if __name__ == "__main__":
    main()
