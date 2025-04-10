"""
Compute topographic prominance of a 1D time series.
Code implemented by OpenAI GPT-4o mini.

Raúl Valenzuela
2025

Returns:
    peaks: indexes of peak values
    prominences: values of computed prominence for each peak
"""

import numpy as np

def find_local_peaks(signal):
    """Find local peaks (maxima) in a 1D array using numpy."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i - 1] < signal[i] > signal[i + 1]:
            peaks.append(i)
    return np.array(peaks)

def compute_prominence_numpy(signal, wlen=None):
    peaks = find_local_peaks(signal)
    prominences = []

    for peak in peaks:
        peak_val = signal[peak]

        # Set window limits
        left_limit = 0 if wlen is None else max(0, peak - wlen)
        right_limit = len(signal) - 1 if wlen is None else min(len(signal) - 1, peak + wlen)

        # ----- Left Side -----
        left_base = left_limit
        for i in range(peak - 1, left_limit - 1, -1):
            if signal[i] > peak_val:  # Hit a higher peak
                break
            left_base = i

        # ----- Right Side -----
        right_base = right_limit
        for i in range(peak + 1, right_limit + 1):
            if signal[i] > peak_val:  # Hit a higher peak
                break
            right_base = i

        # Min values in base intervals
        left_min = np.min(signal[left_base:peak+1]) if left_base < peak else signal[peak]
        right_min = np.min(signal[peak:right_base+1]) if right_base > peak else signal[peak]

        # The higher of the two base values is the "lowest contour line"
        base_val = max(left_min, right_min)
        prominence = peak_val - base_val
        prominences.append(prominence)

    return peaks, np.array(prominences)