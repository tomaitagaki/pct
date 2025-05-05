import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
from dateutil import parser
from scipy.signal import butter, filtfilt, welch
from collections import defaultdict

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, fs, lowcut=4.0, highcut=40.0, order=4):
    """Apply bandpass filter to data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def compute_band_powers(data, fs, window_sec=1, overlap_sec=0):
    """Compute power in different frequency bands using sliding windows
    
    Args:
        data: EEG data array
        fs: sampling frequency
        window_sec: window size in seconds (default: 1s for high temporal resolution)
        overlap_sec: overlap between windows in seconds (default: 0s)
    
    Returns:
        times: center times of windows
        powers: dict containing power arrays for each band
    """
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)
    
    # Create windows
    starts = np.arange(0, len(data) - window_samples, step_samples)
    times = starts / fs + window_sec/2  # Center times of windows
    
    # Initialize power arrays for each band
    powers = {
        'theta': [],  # 4-8 Hz
        'alpha': [],  # 8-12 Hz
        'beta': [],   # 12-30 Hz
        'total': []   # 4-30 Hz (total power in analyzed bands)
    }
    
    for start in starts:
        end = start + window_samples
        window = data[start:end]
        
        # Compute PSD
        freqs, psd = welch(window, fs, nperseg=min(256, len(window)))
        
        # Get power in each band
        theta_mask = (freqs >= 4) & (freqs <= 8)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 12) & (freqs <= 30)
        total_mask = (freqs >= 4) & (freqs <= 30)  # Total power in analyzed range
        
        powers['theta'].append(np.mean(psd[theta_mask]))
        powers['alpha'].append(np.mean(psd[alpha_mask]))
        powers['beta'].append(np.mean(psd[beta_mask]))
        powers['total'].append(np.mean(psd[total_mask]))
    
    # Convert to numpy arrays
    powers = {k: np.array(v) for k, v in powers.items()}
    
    # Add relative alpha power
    powers['relative_alpha'] = powers['alpha'] / powers['total']
    
    return times, powers

def smooth_data(times, data, window_sec=15):
    """Apply sliding window smoothing to data
    
    Args:
        times: array of time points
        data: array of values to smooth
        window_sec: smoothing window in seconds
    
    Returns:
        smoothed_times: center times of smoothing windows
        smoothed_data: smoothed values
    """
    # Convert window from seconds to samples
    dt = times[1] - times[0]  # Time step
    window_samples = int(window_sec / dt)
    
    if window_samples >= len(data):
        return times, data
    
    # Ensure odd window size for centered smoothing
    if window_samples % 2 == 0:
        window_samples += 1
    
    half_window = window_samples // 2
    
    # Create padded data to handle edges
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    
    # Apply sliding window average
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        start = i
        end = i + window_samples
        smoothed_data[i] = np.mean(padded_data[start:end])
    
    return times, smoothed_data

def plot_power_ratio(times, powers, markers_df, ratio_type='alpha_theta', channel_idx=0):
    """Plot power ratio with markers"""
    plt.figure(figsize=(15, 6))
    
    # Compute ratio
    if ratio_type == 'alpha_theta':
        ratio = powers['alpha'] / powers['theta']
        ylabel = 'Alpha/Theta Ratio'
        title = f'Alpha/Theta Ratio Channel {channel_idx} (15s smoothing)'
    else:  # beta_total
        ratio = powers['beta'] / (powers['alpha'] + powers['theta'])
        ylabel = 'Beta/(Alpha+Theta) Ratio'
        title = f'Beta/(Alpha+Theta) Ratio Channel {channel_idx} (15s smoothing)'
    
    # Apply smoothing
    smooth_times, smooth_ratio = smooth_data(times, ratio)
    
    # Plot ratio
    plt.plot(smooth_times, smooth_ratio, 'g-', label='Power Ratio', alpha=0.7)
    
    # Group stimulus markers by stimulus type
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    # Plot shading between pairs of same markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        marker_times = sorted(marker_groups[stim])
        # Shade between pairs
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2,
                          label=f'{stim}' if i == 0 else "")
    
    # Plot marker lines
    ymin, ymax = plt.ylim()
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
            plt.text(marker['seconds'], ymax, str(marker['stimulus']), 
                    rotation=0, ha='right', va='bottom')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_eeg_with_markers_and_shading(eeg_data, eeg_times, markers_df, channel_idx=0):
    """Plot EEG data with markers and shading between stimulus events grouped by stimulus type"""
    plt.figure(figsize=(15, 8))
    plt.plot(eeg_times, eeg_data[:, channel_idx], 'b-', label='EEG', alpha=0.7)
    # Group stimulus markers by stimulus type
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        times = sorted(marker_groups[stim])
        for i in range(0, len(times)-1, 2):
            if i+1 < len(times):
                plt.axvspan(times[i], times[i+1], color=color, alpha=0.2, label=f'{stim}' if i == 0 else "")
    ymin, ymax = plt.ylim()
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
            plt.text(marker['seconds'], ymax, str(marker['stimulus']), rotation=0, ha='right', va='bottom')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.title(f'Filtered EEG Channel {channel_idx} (0.5-40 Hz) with Markers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()

def plot_band_power(times, powers, markers_df, band='alpha', channel_idx=0):
    """Plot band power with markers"""
    plt.figure(figsize=(15, 6))
    
    # Apply smoothing
    smooth_times, smooth_power = smooth_data(times, powers[band])
    
    # Plot power
    plt.plot(smooth_times, smooth_power, 'g-', label=f'{band.capitalize()} Power', alpha=0.7)
    
    # Group stimulus markers by stimulus type
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    # Plot shading between pairs of same markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        marker_times = sorted(marker_groups[stim])
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2, label=f'{stim}' if i == 0 else "")
    
    # Plot marker lines
    ymin, ymax = plt.ylim()
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
            plt.text(marker['seconds'], ymax, str(marker['stimulus']), rotation=0, ha='right', va='bottom')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'{band.capitalize()} Power')
    plt.title(f'{band.capitalize()} Power Channel {channel_idx} (15s smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_relative_power(times, powers, markers_df, band='alpha', channel_idx=0):
    """Plot relative band power with markers"""
    plt.figure(figsize=(15, 6))
    
    # Apply smoothing
    smooth_times, smooth_power = smooth_data(times, powers[f'relative_{band}'])
    
    # Plot relative power
    plt.plot(smooth_times, smooth_power, 'g-', 
             label=f'Relative {band.capitalize()} Power', alpha=0.7)
    
    # Group stimulus markers by stimulus type
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    # Plot shading between pairs of same markers
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        marker_times = sorted(marker_groups[stim])
        # Shade between pairs
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2,
                          label=f'{stim}' if i == 0 else "")
    
    # Plot marker lines
    ymin, ymax = plt.ylim()
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
            plt.text(marker['seconds'], ymax, str(marker['stimulus']), 
                    rotation=0, ha='right', va='bottom')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Relative {band.capitalize()} Power')
    plt.title(f'Relative {band.capitalize()} Power Channel {channel_idx} (15s smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def min_max_scale(data):
    """Min-max scale data to range [0, 1]"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def plot_all_bands(times, powers, markers_df, channel_idx=0):
    """Plot all frequency bands together with markers"""
    plt.figure(figsize=(15, 6))
    
    # Apply smoothing and scaling to each band
    bands = ['alpha', 'beta', 'theta']
    colors = ['g', 'r', 'b']
    smooth_powers = {}
    
    for band in bands:
        # Smooth the data
        smooth_times, smooth_power = smooth_data(times, powers[band])
        # Scale the smoothed data
        scaled_power = min_max_scale(smooth_power)
        smooth_powers[band] = scaled_power
        plt.plot(smooth_times, scaled_power, f'{colors[bands.index(band)]}-', 
                label=f'{band.capitalize()} Power (Scaled)', alpha=0.7)
    
    # Group stimulus markers by stimulus type
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    # Plot shading between pairs of same markers
    shade_colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), shade_colors):
        marker_times = sorted(marker_groups[stim])
        # Shade between pairs
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2,
                          label=f'{stim}' if i == 0 else "")
    
    # Plot marker lines
    ymin, ymax = plt.ylim()
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
            plt.text(marker['seconds'], ymax, str(marker['stimulus']), 
                    rotation=0, ha='right', va='bottom')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Band Power')
    plt.title(f'All Frequency Bands Channel {channel_idx} (15s smoothing, Min-Max Scaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits with a small padding
    plt.ylim(-0.05, 1.05)
    
    return plt.gcf()

def load_latest_data(data_dir='data', file_name=None):
    """Load the most recent EEG and marker data, or use file_name if provided"""
    files = os.listdir(data_dir)
    if file_name:
        base_name = file_name
    else:
        # Find the latest timestamp by normalizing filenames first
        timestamps = set()
        for f in files:
            if f.startswith(('eeg_data_', 'markers_', 'timestamps_')):
                parts = f.split('_')
                if f.startswith('eeg_data_'):
                    ts = '_'.join(parts[2:]).split('.')[0]
                else:
                    ts = parts[1].split('.')[0]
                timestamps.add(ts)
        if not timestamps:
            raise RuntimeError("No data files found in data directory")
        base_name = max(timestamps)
    # Load EEG data
    eeg_data = np.load(os.path.join(data_dir, f'eeg_data_{base_name}.npy'))
    eeg_timestamps = np.load(os.path.join(data_dir, f'timestamps_{base_name}.npy'))
    markers_file = os.path.join(data_dir, f'markers_{base_name}.csv')
    markers_df = pd.read_csv(markers_file)
    if markers_df.empty:
        raise RuntimeError(f"No marker events found in {os.path.basename(markers_file)}")
    
    # Convert marker timestamps to seconds from start
    start_time = parser.parse(markers_df['timestamp'].iloc[0])
    markers_df['seconds'] = [(parser.parse(t) - start_time).total_seconds() 
                            for t in markers_df['timestamp']]
    
    # Convert LSL timestamps to seconds from start
    eeg_start_time = eeg_timestamps[0]
    eeg_times = eeg_timestamps - eeg_start_time
    
    # Calculate sampling rate
    fs = len(eeg_times) / (eeg_times[-1] - eeg_times[0])
    
    # Apply bandpass filter
    print("Applying 0.5-40 Hz bandpass filter...")
    eeg_data = apply_bandpass_filter(eeg_data, fs)
    
    return eeg_data, eeg_times, markers_df, fs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=int, default=0, 
                       help='Channel to plot (default: 0)')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files instead of displaying')
    parser.add_argument('--file_name', type=str, default=None, help='Custom file name for input files (no extension)')
    args = parser.parse_args()
    
    # Load and filter data
    print("Loading data...")
    eeg_data, eeg_times, markers_df, fs = load_latest_data(file_name=args.file_name)
    
    print("\nData summary:")
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Recording duration: {eeg_times[-1]:.1f} seconds")
    print(f"Sampling rate: {fs:.1f} Hz")
    print(f"Number of markers: {len(markers_df)}")
    
    # Create plots
    print("\nCreating plots...")
    
    # Plot 1: Filtered EEG with shading
    fig1 = plot_eeg_with_markers_and_shading(eeg_data, eeg_times, markers_df, args.channel)
    
    # Compute band powers for all plots
    times, powers = compute_band_powers(eeg_data[:, args.channel], fs)
    
    # Plot 2: Alpha power
    fig2 = plot_band_power(times, powers, markers_df, 'alpha', args.channel)
    
    # Plot 3: Alpha/Theta ratio
    fig3 = plot_power_ratio(times, powers, markers_df, 'alpha_theta', args.channel)
    
    # Plot 4: Beta/(Alpha+Theta) ratio
    fig4 = plot_power_ratio(times, powers, markers_df, 'beta_total', args.channel)
    
    # Plot 5: Relative alpha power
    fig5 = plot_relative_power(times, powers, markers_df, 'alpha', args.channel)
    
    # Plot 6: All frequency bands
    fig6 = plot_all_bands(times, powers, markers_df, args.channel)
    
    # Save or show plots
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save filtered EEG plot
        filename1 = f'eeg_filtered_plot_{timestamp}.png'
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        # Save alpha power plot
        filename2 = f'alpha_power_plot_{timestamp}.png'
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        # Save alpha/theta ratio plot
        filename3 = f'alpha_theta_ratio_{timestamp}.png'
        fig3.savefig(filename3, dpi=300, bbox_inches='tight')
        # Save beta/(alpha+theta) ratio plot
        filename4 = f'beta_total_ratio_{timestamp}.png'
        fig4.savefig(filename4, dpi=300, bbox_inches='tight')
        # Save relative alpha power plot
        filename5 = f'relative_alpha_power_{timestamp}.png'
        fig5.savefig(filename5, dpi=300, bbox_inches='tight')
        # Save all bands plot
        filename6 = f'all_bands_{timestamp}.png'
        fig6.savefig(filename6, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved as: {filename1}, {filename2}, {filename3}, {filename4}, {filename5}, and {filename6}")
    else:
        plt.show()

if __name__ == "__main__":
    main() 