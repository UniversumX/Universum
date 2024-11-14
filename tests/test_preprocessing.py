import numpy as np
import pandas as pd
from ModelDevelopment.preprocessing import plot_fft, time_align_accel_data_by_linearly_interpolating

def test_plot_fft():
    # Sample data to test
    data = np.random.randn(1000)
    sampling_frequency = 256
    fft_window_size = 128
    percent_overlap_between_windows = 0.5
    
    # Call the function and check the output shape
    pxx, freqs, bins = plot_fft(data, sampling_frequency, fft_window_size, percent_overlap_between_windows)
    assert pxx.shape[0] > 0, "pxx should contain data"
    assert len(freqs) > 0, "Frequency data should be non-empty"
    assert len(bins) > 0, "Bins should be non-empty"

def test_time_align_accel_data_by_linearly_interpolating():
    # Dummy accelerometer and eeg data for testing
    accel_data = pd.DataFrame({'timestamp': [0, 1, 2], 'x': [0.1, 0.2, 0.3]})
    eeg_data = pd.DataFrame({'timestamp': [0, 0.5, 1.5, 2.5]})
    
    aligned_data = time_align_accel_data_by_linearly_interpolating(accel_data, eeg_data)
    assert aligned_data.shape[0] == len(eeg_data), "Output rows should match EEG data rows"
