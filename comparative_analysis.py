import os
import numpy as np
from slope_entropy import compute_slope_entropy

DATA_FILE = os.path.join("processed_data", "bearing1_hybrid_dataset.npz")

def compute_metrics(signal):
    # Quantitative metrics
    rms = np.sqrt(np.mean(signal**2))
    p2p = np.max(signal) - np.min(signal)
    
    mean = np.mean(signal)
    std = np.std(signal)
    kurt = np.mean((signal - mean)**4) / (std**4) if std > 0 else 0.0
    
    # Average slope entropy over windows of 2048
    window_size = 2048
    n_windows = len(signal) // window_size
    entropies = []
    for w in range(n_windows):
        window = signal[w*window_size : (w+1)*window_size]
        e = compute_slope_entropy(window, m=3, gamma=1.0, delta=0.1)
        entropies.append(e)
        
    slpen = np.mean(entropies)
    
    return rms, p2p, kurt, slpen

def main():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found.")
        return
        
    dataset = np.load(DATA_FILE)
    data = dataset['data']
    timestamps = dataset['timestamps']
    
    # Define indices for the 3 stages based on our previous findings
    healthy_idx = 0        # 2004.02.15.00.02.39
    transition_idx = 42    # 2004.02.17.08.02.39 (Turning Point)
    faulty_idx = 71        # 2004.02.18.22.42.39 (Deep degradation)
    
    stages = [
        ("Healthy", healthy_idx),
        ("Transition", transition_idx),
        ("Faulty", faulty_idx)
    ]
    
    print(f"{'Stage':<15} | {'Timestamp':<20} | {'RMS (g)':<10} | {'P2P (g)':<10} | {'Kurtosis':<10} | {'SlpEn':<10}")
    print("-" * 85)
    
    for stage_name, idx in stages:
        signal = data[idx]
        timestamp = timestamps[idx]
        
        rms, p2p, kurt, slpen = compute_metrics(signal)
        
        print(f"{stage_name:<15} | {timestamp:<20} | {rms:<10.4f} | {p2p:<10.4f} | {kurt:<10.4f} | {slpen:<10.4f}")

if __name__ == "__main__":
    main()
