import os
import numpy as np
import matplotlib.pyplot as plt

# Directorios de datos y salida
DATA_DIR = "2nd_test"
LATEX_DIR = "latex"

# Archivos representativos a comparar
HEALTHY_FILE = os.path.join(DATA_DIR, "2004.02.12.10.32.39") # Sano
DEGRADED_FILE = os.path.join(DATA_DIR, "2004.02.18.23.52.39") # Defectuoso

SAMPLING_RATE = 20000.0  # Frecuencia de muestreo (20 kHz)
BEARING_COL = 0          # Rodamiento 1 (Canal 1)

def load_and_stats(filepath):
    
    signal = np.loadtxt(filepath, delimiter="\t", usecols=(BEARING_COL,), dtype=np.float32)
    
    # metricas básicas
    rms = np.sqrt(np.mean(signal**2))
    p2p = np.max(signal) - np.min(signal)
    
    # curtosis
    mean = np.mean(signal)
    std = np.std(signal)
    kurt = np.mean((signal - mean)**4) / (std**4) if std > 0 else 0.0
    
    return signal, rms, p2p, kurt

def plot_and_save(signal, label, filename, color):
    n_samples = len(signal)
    time = np.arange(n_samples) / SAMPLING_RATE
    
    # Graficar los primeros 0.1 segundos (3 rotaciones completas del eje a 2000 RPM)
    zoom_samples = int(0.1 * SAMPLING_RATE)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time[:zoom_samples], signal[:zoom_samples], color=color, lw=1)
    plt.title(f"Vibration Signal - Bearing 1 ({label})", fontsize=12, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=10)
    plt.ylabel("Acceleration (g)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, 0.1)
    
    
    if "Healthy" in label or "Saludable" in label:
        plt.ylim(-1.0, 1.0)
    else:
        plt.ylim(-1.0, 1.0)
        
    plt.tight_layout()
    os.makedirs(LATEX_DIR, exist_ok=True)
    
    filepath = os.path.join(LATEX_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    print("=== DATA EXPLORATION AND ANALYSIS ===")
    
    # Procesar señal saludable
    sig_h, rms_h, p2p_h, kurt_h = load_and_stats(HEALTHY_FILE)
    print(f"Healthy Signal ({os.path.basename(HEALTHY_FILE)}):")
    print(f"  RMS: {rms_h:.4f} g | Peak-to-Peak: {p2p_h:.4f} g | Kurtosis: {kurt_h:.4f}\n")
    plot_and_save(sig_h, "Healthy Stage", "raw_healthy_signal.png", "#1f77b4")
    
    # Procesar señal degradada
    sig_d, rms_d, p2p_d, kurt_d = load_and_stats(DEGRADED_FILE)
    print(f"Degraded Signal ({os.path.basename(DEGRADED_FILE)}):")
    print(f"  RMS: {rms_d:.4f} g | Peak-to-Peak: {p2p_d:.4f} g | Kurtosis: {kurt_d:.4f}\n")
    plot_and_save(sig_d, "Degraded Stage", "raw_degraded_signal.png", "#d62728")

if __name__ == "__main__":
    main()
