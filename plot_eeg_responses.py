import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime
import dateutil.parser
from scipy import signal
import argparse
from collections import defaultdict
import traceback

def convert_marker_timestamp(timestamp, start_time):
    """Convert timestamp to seconds from start, handling both ISO strings and numeric values"""
    if isinstance(timestamp, (float, np.float64, int)):
        return float(timestamp)  # Already in seconds
    try:
        # Try parsing as ISO timestamp
        dt = dateutil.parser.isoparse(timestamp)
        # Convert to UTC timestamp in seconds
        utc_seconds = dt.timestamp()
        # Return relative time from start
        return utc_seconds - start_time
    except (TypeError, ValueError):
        print(f"Warning: Could not parse timestamp {timestamp}, using as is")
        return float(timestamp)

def apply_bandpass_filter(data, sfreq, lowcut=4, highcut=40, order=4):
    """Apply a bandpass filter to the EEG data.
    
    Args:
        data: EEG data array (channels x samples)
        sfreq: Sampling frequency in Hz
        lowcut: Lower frequency bound in Hz
        highcut: Upper frequency bound in Hz
        order: Filter order
    
    Returns:
        Filtered EEG data array
    """
    nyquist = sfreq / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    
    return filtered_data

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
    
    if eeg_data.shape[0] > eeg_data.shape[1]:
        print("Transposing EEG data to correct shape...")
        eeg_data = eeg_data.T
    
    # Make EEG timestamps relative to start
    eeg_start_time = eeg_timestamps[0]
    eeg_timestamps = eeg_timestamps - eeg_start_time
    
    print(f"\nData validation:")
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Number of channels: {eeg_data.shape[0]}")
    print(f"Number of samples: {eeg_data.shape[1]}")
    print(f"EEG duration: {eeg_timestamps[-1] - eeg_timestamps[0]:.1f} seconds")
    
    # Calculate sampling rate
    sfreq = eeg_data.shape[1]/(eeg_timestamps[-1] - eeg_timestamps[0])
    print(f"Sampling rate: {sfreq:.1f} Hz")
    
    # Apply bandpass filter
    print("\nApplying 4-40 Hz bandpass filter...")
    eeg_data = apply_bandpass_filter(eeg_data, sfreq)
    print("Filtering complete")
    
    # Convert marker timestamps to seconds from start
    if len(markers_df) > 0:
        try:
            # Get the first marker timestamp as reference
            first_timestamp = markers_df['timestamp'].iloc[0]
            
            if isinstance(first_timestamp, str):
                try:
                    # Parse timestamps and convert to UTC seconds
                    markers_df['seconds'] = markers_df['timestamp'].apply(
                        lambda x: dateutil.parser.isoparse(x).timestamp() if pd.notna(x) and str(x).strip() else None
                    )
                    # Make relative to start
                    markers_df['seconds'] = markers_df['seconds'] - markers_df['seconds'].min()
                    print(f"\nConverted ISO timestamps to relative seconds")
                except (TypeError, ValueError) as e:
                    print(f"Warning: Could not parse timestamps as ISO format: {e}")
                    # Assume timestamps are already in seconds
                    markers_df['seconds'] = pd.to_numeric(markers_df['timestamp'], errors='coerce')
                    markers_df['seconds'] = markers_df['seconds'] - markers_df['seconds'].min()
            else:
                # Numeric timestamps - make relative to start
                markers_df['seconds'] = pd.to_numeric(markers_df['timestamp'], errors='coerce')
                markers_df['seconds'] = markers_df['seconds'] - markers_df['seconds'].min()
            
            # Drop any rows with invalid timestamps
            markers_df = markers_df.dropna(subset=['seconds'])
            
            print(f"\nMarker timing (relative to recording start):")
            print(f"First marker: {markers_df['seconds'].min():.1f} seconds")
            print(f"Last marker: {markers_df['seconds'].max():.1f} seconds")
            print(f"Total recording duration: {markers_df['seconds'].max() - markers_df['seconds'].min():.1f} seconds")
        except Exception as e:
            print(f"Error processing timestamps: {e}")
            print("Marker DataFrame head:")
            print(markers_df.head())
            raise
    
    print(f"\nMarker types:")
    if 'event_type' in markers_df.columns:
        print(markers_df['event_type'].value_counts())
    else:
        print("No event_type column found in markers")
    
    return eeg_data, eeg_timestamps, markers_df

def plot_eeg_with_markers(eeg_data, timestamps, markers_df, channel=0):
    """Plot EEG data with all marker types"""
    if channel >= eeg_data.shape[0]:
        print(f"WARNING: Requested channel {channel} but only {eeg_data.shape[0]} channels available")
        channel = 0
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, eeg_data[channel], 'b-', label='EEG (4-40 Hz)', alpha=0.7)
    print(f"\nPlotting details:")
    print(f"Plotting channel {channel} of {eeg_data.shape[0]} (4-40 Hz filtered)")
    print(f"Time range: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
    print(f"EEG amplitude range: {np.min(eeg_data[channel]):.1f} to {np.max(eeg_data[channel]):.1f} µV")
    marker_styles = {
        'game_start': {'color': 'blue', 'linestyle': '-', 'alpha': 0.5, 'label': 'Game Start'},
        'game_end': {'color': 'red', 'linestyle': '-', 'alpha': 0.5, 'label': 'Game End'},
        'round_start': {'color': 'purple', 'linestyle': '-', 'alpha': 0.3, 'label': 'Round Start'},
        'stimulus': {
            'triangle': {'color': 'orange', 'linestyle': '-', 'alpha': 0.3, 'label': 'Triangle'},
            'square': {'color': 'green', 'linestyle': '-', 'alpha': 0.3, 'label': 'Square'},
            'circle': {'color': 'red', 'linestyle': '-', 'alpha': 0.3, 'label': 'Circle'}
        },
        'response': {
            'A': {'color': 'g', 'linestyle': '--', 'alpha': 0.5, 'label': 'Response A'},
            'L': {'color': 'r', 'linestyle': '--', 'alpha': 0.5, 'label': 'Response L'}
        }
    }
    used_labels = set()
    event_counts = {
        'game_start': 0, 
        'game_end': 0,
        'round_start': 0, 
        'stimulus_triangle': 0,
        'stimulus_square': 0,
        'stimulus_circle': 0,
        'response_A': 0, 
        'response_L': 0
    }
    y_min, y_max = plt.ylim()
    text_top = y_max + (y_max - y_min) * 0.05
    text_bottom = y_min - (y_max - y_min) * 0.05
    for _, marker in markers_df.iterrows():
        event_type = marker['event_type']
        if event_type in ['game_start', 'game_end']:
            style = marker_styles[event_type]
            label = style['label'] if event_type not in used_labels else None
            plt.axvline(x=marker['seconds'], color=style['color'], 
                       linestyle=style['linestyle'], alpha=style['alpha'], label=label)
            if label:
                used_labels.add(event_type)
            event_counts[event_type] += 1
            plt.text(marker['seconds'], text_top, event_type.replace('_', ' ').title(), 
                    rotation=90, verticalalignment='bottom')
        elif event_type == 'round_start':
            style = marker_styles['round_start']
            label = style['label'] if 'round_start' not in used_labels else None
            plt.axvline(x=marker['seconds'], color=style['color'], 
                       linestyle=style['linestyle'], alpha=style['alpha'], label=label)
            if label:
                used_labels.add('round_start')
            event_counts['round_start'] += 1
            plt.text(marker['seconds'], text_top, f"Round {marker['round']}", 
                    rotation=90, verticalalignment='bottom')
        elif event_type == 'stimulus' and pd.notna(marker['stimulus']):
            stim_type = str(marker['stimulus']).lower()
            if stim_type in marker_styles['stimulus']:
                style = marker_styles['stimulus'][stim_type]
                label = style['label'] if f'stimulus_{stim_type}' not in used_labels else None
                plt.axvline(x=marker['seconds'], color=style['color'], 
                           linestyle=style['linestyle'], alpha=style['alpha'], label=label)
                if label:
                    used_labels.add(f'stimulus_{stim_type}')
                event_counts[f'stimulus_{stim_type}'] += 1
        elif event_type == 'response' and pd.notna(marker['response']):
            response = str(marker['response']).upper()
            if response in marker_styles['response']:
                style = marker_styles['response'][response]
                label = style['label'] if f'response_{response}' not in used_labels else None
                plt.axvline(x=marker['seconds'], color=style['color'], 
                           linestyle=style['linestyle'], alpha=style['alpha'], label=label)
                if label:
                    used_labels.add(f'response_{response}')
                event_counts[f'response_{response}'] += 1
                if pd.notna(marker['correct']):
                    correct = str(marker['correct']).lower() == 'true'
                    plt.text(marker['seconds'], text_bottom, 
                            f"{response} ({'✓' if correct else '✗'})", 
                            rotation=90, verticalalignment='top')
    print(f"\nEvents plotted:")
    print(f"Game starts: {event_counts['game_start']}")
    print(f"Game ends: {event_counts['game_end']}")
    print(f"Round starts: {event_counts['round_start']}")
    print(f"Stimuli shown:")
    print(f"  Triangle: {event_counts['stimulus_triangle']}")
    print(f"  Square: {event_counts['stimulus_square']}")
    print(f"  Circle: {event_counts['stimulus_circle']}")
    print(f"Responses:")
    print(f"  A: {event_counts['response_A']}")
    print(f"  L: {event_counts['response_L']}")
    
    # Customize plot
    plt.title(f'EEG Data (Channel {channel}) with Event Markers')
    plt.xlabel('Time (seconds from recording start)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('data', f'eeg_plot_{timestamp}.png')
    plt.savefig(save_path)
    print(f"\nPlot saved as: {save_path}")
    
    plt.show()

def compute_response_locked_averages(eeg_data, timestamps, markers_df, channel=0, pre_time=0.5, post_time=2.0):
    """Compute response-locked averages for correct and incorrect responses.
    
    Args:
        eeg_data: EEG data array (channels x samples)
        timestamps: Array of timestamps
        markers_df: DataFrame containing markers
        channel: Channel to analyze
        pre_time: Time before response (in seconds)
        post_time: Time after response (in seconds)
    """
    # Calculate sampling rate
    sfreq = 1 / np.mean(np.diff(timestamps))
    pre_samples = int(pre_time * sfreq)
    post_samples = int(post_time * sfreq)
    epoch_length = pre_samples + post_samples
    
    # Initialize lists to store epochs
    correct_epochs = []
    incorrect_epochs = []
    
    # Get response events
    response_events = markers_df[markers_df['event_type'] == 'response'].copy()
    response_events['correct'] = response_events['correct'].map(lambda x: str(x).lower() == 'true')
    
    for _, event in response_events.iterrows():
        try:
            # Find the closest timestamp index
            response_time = event['seconds']
            response_idx = np.argmin(np.abs(timestamps - response_time))
            
            # Extract epoch
            start_idx = response_idx - pre_samples
            end_idx = response_idx + post_samples
            
            if start_idx >= 0 and end_idx < eeg_data.shape[1]:
                epoch = eeg_data[channel, start_idx:end_idx]
                if event['correct']:
                    correct_epochs.append(epoch)
                else:
                    incorrect_epochs.append(epoch)
        except Exception as e:
            print(f"Error processing epoch: {e}")
            continue
    
    # Convert to arrays
    correct_epochs = np.array(correct_epochs) if correct_epochs else np.array([])
    incorrect_epochs = np.array(incorrect_epochs) if incorrect_epochs else np.array([])
    
    # Create time axis for plotting
    epoch_times = np.linspace(-pre_time, post_time, epoch_length)
    
    # Compute averages and standard errors
    correct_avg = np.mean(correct_epochs, axis=0) if len(correct_epochs) > 0 else None
    correct_sem = np.std(correct_epochs, axis=0) / np.sqrt(len(correct_epochs)) if len(correct_epochs) > 0 else None
    incorrect_avg = np.mean(incorrect_epochs, axis=0) if len(incorrect_epochs) > 0 else None
    incorrect_sem = np.std(incorrect_epochs, axis=0) / np.sqrt(len(incorrect_epochs)) if len(incorrect_epochs) > 0 else None
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    if correct_avg is not None:
        plt.plot(epoch_times, correct_avg, 'g-', label=f'Correct (n={len(correct_epochs)})')
        plt.fill_between(epoch_times, 
                        correct_avg - correct_sem,
                        correct_avg + correct_sem,
                        color='g', alpha=0.2)
    
    if incorrect_avg is not None:
        plt.plot(epoch_times, incorrect_avg, 'r-', label=f'Incorrect (n={len(incorrect_epochs)})')
        plt.fill_between(epoch_times, 
                        incorrect_avg - incorrect_sem,
                        incorrect_avg + incorrect_sem,
                        color='r', alpha=0.2)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.title(f'Response-locked Average (Channel {channel})')
    plt.xlabel('Time relative to response (seconds)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('data', f'response_locked_average_{timestamp}.png')
    plt.savefig(save_path)
    print(f"\nResponse-locked average plot saved as: {save_path}")
    
    plt.show()
    
    return {
        'correct_epochs': correct_epochs,
        'incorrect_epochs': incorrect_epochs,
        'epoch_times': epoch_times
    }

def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def smooth_data(times, data, window_sec=15):
    dt = times[1] - times[0]
    window_samples = int(window_sec / dt)
    if window_samples >= len(data):
        return times, data
    if window_samples % 2 == 0:
        window_samples += 1
    half_window = window_samples // 2
    padded_data = np.pad(data, (half_window, half_window), mode='edge')
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        start = i
        end = i + window_samples
        smoothed_data[i] = np.mean(padded_data[start:end])
    return times, smoothed_data

def compute_band_powers(data, fs, window_sec=1, overlap_sec=0):
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)
    starts = np.arange(0, len(data) - window_samples, step_samples)
    times = starts / fs + window_sec/2
    powers = {
        'theta': [],  # 4-8 Hz
        'alpha': [],  # 8-12 Hz
        'beta': []    # 12-30 Hz
    }
    for start in starts:
        end = start + window_samples
        window = data[start:end]
        freqs, psd = signal.welch(window, fs, nperseg=min(256, len(window)))
        theta_mask = (freqs >= 4) & (freqs <= 8)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 12) & (freqs <= 30)
        powers['theta'].append(np.mean(psd[theta_mask]))
        powers['alpha'].append(np.mean(psd[alpha_mask]))
        powers['beta'].append(np.mean(psd[beta_mask]))
    powers = {k: np.array(v) for k, v in powers.items()}
    return times, powers

def plot_band_power(times, powers, markers_df, band='alpha', channel_idx=0, save_path=None):
    plt.figure(figsize=(15, 6))
    smooth_times, smooth_power = smooth_data(times, powers[band])
    scaled_power = min_max_scale(smooth_power)
    plt.plot(smooth_times, scaled_power, label=f'{band.capitalize()} Power (Scaled)', alpha=0.7)
    
    # Add stimulus markers and regions
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        marker_times = sorted(marker_groups[stim])
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2, label=f'{stim}' if i == 0 else "")
    
    # Add round start markers
    round_starts = markers_df[markers_df['event_type'] == 'round_start']
    for _, marker in round_starts.iterrows():
        plt.axvline(x=marker['seconds'], color='purple', linestyle='-', alpha=0.5, label='Round Start' if _ == round_starts.index[0] else "")
        plt.text(marker['seconds'], 1.05, f"Round {marker['round']}", rotation=90, verticalalignment='bottom')
    
    # Add stimulus markers
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'{band.capitalize()} Power (min-max scaled)')
    plt.title(f'{band.capitalize()} Power Channel {channel_idx} (15s smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {band} power plot: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_relative_alpha_power(eeg_data, timestamps, markers_df, channel=0, window_sec=1, overlap_sec=0, smoothing_sec=15, save=False, file_name=None):
    """Plot relative alpha power (alpha/(theta+beta)) with stimulus markers."""
    fs = 1 / np.mean(np.diff(timestamps))
    times, powers = compute_band_powers(eeg_data[channel], fs, window_sec, overlap_sec)
    
    # Compute relative alpha power
    relative_alpha = powers['alpha'] / (powers['theta'] + powers['beta'])
    smooth_times, smooth_power = smooth_data(times, relative_alpha, smoothing_sec)
    scaled_power = min_max_scale(smooth_power)
    
    plt.figure(figsize=(15, 6))
    plt.plot(smooth_times, scaled_power, label='Relative Alpha Power (Scaled)', alpha=0.7)
    
    # Add stimulus markers and regions
    marker_groups = defaultdict(list)
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            marker_groups[marker['stimulus']].append(marker['seconds'])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(marker_groups)))
    for stim, color in zip(sorted(marker_groups.keys()), colors):
        marker_times = sorted(marker_groups[stim])
        for i in range(0, len(marker_times)-1, 2):
            if i+1 < len(marker_times):
                plt.axvspan(marker_times[i], marker_times[i+1], color=color, alpha=0.2, label=f'{stim}' if i == 0 else "")
    
    # Add round start markers
    round_starts = markers_df[markers_df['event_type'] == 'round_start']
    for _, marker in round_starts.iterrows():
        plt.axvline(x=marker['seconds'], color='purple', linestyle='-', alpha=0.5, label='Round Start' if _ == round_starts.index[0] else "")
        plt.text(marker['seconds'], 1.05, f"Round {marker['round']}", rotation=90, verticalalignment='bottom')
    
    # Add stimulus markers
    for _, marker in markers_df.iterrows():
        if marker.get('event_type') == 'stimulus' and pd.notnull(marker.get('stimulus')):
            plt.axvline(x=marker['seconds'], color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Relative Alpha Power (min-max scaled)')
    plt.title(f'Relative Alpha Power Channel {channel} ({smoothing_sec}s smoothing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    if save and file_name:
        plt.savefig(f'relative_alpha_power_{file_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=int, default=0, help='Channel to plot (default: 0)')
    parser.add_argument('--save', action='store_true', help='Save plot to file instead of displaying')
    parser.add_argument('--file_name', type=str, default=None, help='Custom file name for input files (no extension)')
    args = parser.parse_args()
    # Load data
    print("Loading data...")
    eeg_data, eeg_times, markers_df = load_latest_data(file_name=args.file_name)
    
    # Plot full EEG with markers
    plot_eeg_with_markers(eeg_data, eeg_times, markers_df, channel=args.channel)
    
    # Compute and plot response-locked averages
    compute_response_locked_averages(eeg_data, eeg_times, markers_df, channel=args.channel)

    plot_relative_alpha_power(eeg_data, eeg_times, markers_df, channel=args.channel, save=args.save, file_name=args.file_name)
    
    # --- Add band power plots ---
    fs = 1 / np.mean(np.diff(eeg_times))
    times, powers = compute_band_powers(eeg_data[args.channel], fs)
    base_name = args.file_name if args.file_name else 'latest'
    for band in ['alpha', 'beta', 'theta']:
        save_path = f'data/{band}_power_{base_name}.png' if args.save else None
        plot_band_power(times, powers, markers_df, band=band, channel_idx=args.channel, save_path=save_path)

if __name__ == "__main__":
    main() 