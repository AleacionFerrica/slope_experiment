import numpy as np

def compute_slope_entropy(signal, m=3, gamma=1.0, delta=0.1):
    """
    Parameters:
    
    signal : numpy.ndarray
        The input 1D time series array.
    m      : int
        The embedding dimension (length of the subsequence). Default is 3.
    gamma  : float
        The first threshold parameter. Default is 1.0.
    delta  : float
        The second threshold parameter. Default is 0.1.
        
    Returns: The computed slope entropy value.
    """
    N = len(signal)
    if N < m:
        return 0.0

    # Z-score normalization 
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val > 0:
        y = (signal - mean_val) / std_val
    else:
        y = signal - mean_val

    # Compute differences between consecutive samples
    # diffs[i] = y[i+1] - y[i]
    diffs = np.diff(y)

    # Apply thresholding rules to map differences to symbols 
    symbols = np.zeros_like(diffs, dtype=np.int8)
    
    # Rule 1: xi > xi-1 + gamma  -->  diff > gamma
    symbols[diffs > gamma] = 2
    
    # Rule 2: xi > xi-1 + delta AND xi <= xi-1 + gamma --> delta < diff <= gamma
    symbols[(diffs > delta) & (diffs <= gamma)] = 1
    
    # Rule 3: |xi - xi-1| <= delta --> -delta <= diff <= delta
    symbols[np.abs(diffs) <= delta] = 0
    
    # Rule 4: xi < xi-1 - delta AND xi >= xi-1 - gamma --> -gamma <= diff < -delta
    symbols[(diffs < -delta) & (diffs >= -gamma)] = -1
    
    # Rule 5: xi < xi-1 - gamma --> diff < -gamma
    symbols[diffs < -gamma] = -2

    # Form feature vectors (patterns) of length m-1
    
    pattern_length = m - 1 # Since symbols array has length N-1, we need rolling windows of length m-1
    num_patterns = len(symbols) - pattern_length + 1
    
    if num_patterns <= 0:
        return 0.0

    # Efficient rolling window trick using strides
    shape = (num_patterns, pattern_length)
    strides = (symbols.strides[0], symbols.strides[0])
    patterns = np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)

    # Count occurrences of each unique pattern
    # We can uniquely identify each pattern by converting it to a tuple or a unique integer
    # Since symbols are in [-2, 2], we can shift them to [0, 4] and treat as base-5 integers
    shifted_patterns = patterns + 2
    base = 5
    
    # Compute base-5 representation for each pattern to get a unique 1D ID
    powers_of_base = base ** np.arange(pattern_length - 1, -1, -1)
    pattern_ids = np.dot(shifted_patterns, powers_of_base)

    # Count frequencies of each unique ID
    _, counts = np.unique(pattern_ids, return_counts=True)

    # Compute pseudo-probabilities and Shannon entropy
    total_extracted_patterns = num_patterns
    probabilities = counts / total_extracted_patterns
    
    # Filter out zero probabilities (though np.unique ensures counts > 0)
    p = probabilities[probabilities > 0]
    
    # Slope Entropy calculation 
    slpen = -np.sum(p * np.log(p))
    
    return slpen

if __name__ == "__main__":
 
    y_test = np.array([8, 42, 15, 9, 6, 6], dtype=float)
    entropy_val = compute_slope_entropy(y_test, m=2, gamma=1.0, delta=0.1)
    
    print(f"Test Signal: {y_test}")
    print(f"Computed Slope Entropy: {entropy_val:.4f}")

    
