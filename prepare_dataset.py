import os
import glob
import numpy as np


DATA_DIR = "2nd_test"
OUTPUT_DIR = "processed_data"

# Parámetros del subset
START_DATE = "2004.02.15.00.02.39"  # Inicio de la ventana (15 de febrero)
END_DATE = "2004.02.18.23.52.39"    # Fin de la ventana (18 de febrero). 
STEP = 8                            # Muestreo periódico (1 de cada 8 archivos)
BEARING_COL = 0                     # Columna del Rodamiento 1 (Canal 1 es índice 0)
SAMPLES_PER_FILE = 20480            # Número de muestras por archivo (1 segundo a 20 kHz)

def main():
    print("--- INICIANDO PROCESAMIENTO Y COMPRESIÓN DE DATOS ---")
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "2004.*")))
    if not all_files:
        print(f"Error: No se encontraron archivos de datos en '{DATA_DIR}'. Verifica la ruta.")
        return

    # Filtrar los archivos por rango de fechas
    selected_files = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        if START_DATE <= filename <= END_DATE:
            selected_files.append(filepath)

    print(f"Archivos dentro del rango: {len(selected_files)}")

    # muestreo periódico paso k = 8
    sampled_files = selected_files[::STEP]
    print(f"Archivos finales: {len(sampled_files)}")

    
    combined_data = []
    timestamps = []

    # Procesar cada archivo
    for i, filepath in enumerate(sampled_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(sampled_files)}] Procesando {filename}...", end="\r")
        
        try:
            # Leer el archivo de texto
            data = np.loadtxt(filepath, delimiter="\t", usecols=(BEARING_COL,), dtype=np.float32)# float32 para ahorrar memoria y almacenamiento
            
            # Verificar integridad 
            if len(data) != SAMPLES_PER_FILE:
                data = np.resize(data, SAMPLES_PER_FILE)

            combined_data.append(data)
            timestamps.append(filename)

        except Exception as e:
            print(f"\nError al procesar el archivo {filename}: {e}")



    # Guardar archivo combinado y comprimido 
    if combined_data:
        combined_data_arr = np.array(combined_data, dtype=np.float32)
        timestamps_arr = np.array(timestamps)

        # Guardar en formato .npz (comprimido nativo de NumPy)
        npz_filepath = os.path.join(OUTPUT_DIR, "bearing1_hybrid_dataset.npz")
        np.savez_compressed(npz_filepath, data=combined_data_arr, timestamps=timestamps_arr)
        
        print("\n--- RESUMEN DELsubset CREADO ---")
       
        print(f" Dataset único combinado guardado en: {npz_filepath}")
        print(f"   - Dimensiones de los datos: {combined_data_arr.shape} (N_archivos, N_muestras)")
        print(f"   - Tamaño total en disco: {os.path.getsize(npz_filepath) / (1024*1024):.2f} MB")
        print("------------------------------------------------")
    else:
        print("No se pudo generar el archivo combinado.")

if __name__ == "__main__":
    main()
