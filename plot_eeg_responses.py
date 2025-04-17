import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime
import dateutil.parser

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

def load_latest_data(data_dir='data'):
    """Load the most recent data files and convert timestamps to consistent units"""
    eeg_files = glob.glob(os.path.join(data_dir, 'eeg_data_*.npy'))
    timestamp_files = glob.glob(os.path.join(data_dir, 'timestamps_*.npy'))
    marker_files = glob.glob(os.path.join(data_dir, 'markers_*.csv'))
    
    if not (eeg_files and timestamp_files and marker_files):
        raise FileNotFoundError("No data files found")
    
    eeg_files.sort()
    timestamp_files.sort()
    marker_files.sort()
    
    print('\nFound data files:')
    print(f'EEG file: {eeg_files[-1]}')
    print(f'Timestamp file: {timestamp_files[-1]}')
    print(f'Marker file: {marker_files[-1]}')

    # Load data
    eeg_data = np.load(eeg_files[-1])  # Units: microvolts (µV)
    eeg_timestamps = np.load(timestamp_files[-1])  # Units: seconds since stream start
    
    try:
        # Read markers CSV with explicit column names
        expected_columns = ['timestamp', 'event_type', 'stimulus', 'response', 'correct', 'accuracy', 'round']
        markers_df = pd.read_csv(marker_files[-1], 
                               on_bad_lines='skip',
                               dtype={col: str for col in expected_columns})  # Read all as strings initially
        
        print(f"\nLoaded {len(markers_df)} markers from CSV")
        print("Marker columns:", markers_df.columns.tolist())
        
        # Verify we have the timestamp column
        if 'timestamp' not in markers_df.columns:
            print("\nWARNING: CSV format incorrect. Attempting to fix...")
            # Try reading without headers if the first row might be data
            markers_df = pd.read_csv(marker_files[-1], 
                                   names=expected_columns,
                                   header=None,
                                   on_bad_lines='skip',
                                   dtype={col: str for col in expected_columns})
        
        print("\nMarker data sample:")
        print(markers_df.head())
        
    except Exception as e:
        print(f"Error reading markers CSV: {e}")
        raise
    
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
    print(f"Sampling rate: {eeg_data.shape[1]/(eeg_timestamps[-1] - eeg_timestamps[0]):.1f} Hz")
    
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
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot EEG data
    plt.plot(timestamps, eeg_data[channel], 'b-', label='EEG', alpha=0.7)
    print(f"\nPlotting details:")
    print(f"Plotting channel {channel} of {eeg_data.shape[0]}")
    print(f"Time range: {timestamps[0]:.1f}s to {timestamps[-1]:.1f}s")
    print(f"EEG amplitude range: {np.min(eeg_data[channel]):.1f} to {np.max(eeg_data[channel]):.1f} µV")
    
    # Define marker styles for different event types
    marker_styles = {
        'round_start': {'color': 'purple', 'linestyle': '-', 'alpha': 0.3, 'label': 'Round Start'},
        'stimulus': {'color': 'orange', 'linestyle': '-', 'alpha': 0.3, 'label': 'Stimulus Shown'},
        'response': {
            'A': {'color': 'g', 'linestyle': '--', 'alpha': 0.5, 'label': 'Response A'},
            'L': {'color': 'r', 'linestyle': '--', 'alpha': 0.5, 'label': 'Response L'}
        }
    }
    
    used_labels = set()
    event_counts = {'round_start': 0, 'stimulus': 0, 'response_A': 0, 'response_L': 0}
    
    # Get plot y-limits for text placement
    y_min, y_max = plt.ylim()
    text_top = y_max + (y_max - y_min) * 0.05  # 5% above the top
    text_bottom = y_min - (y_max - y_min) * 0.05  # 5% below the bottom
    
    for _, marker in markers_df.iterrows():
        # Print marker data for debugging
        print(f"Processing marker: {marker.to_dict()}")
        
        # Check for round start
        if marker['event_type'] == 'round_start':
            style = marker_styles['round_start']
            label = style['label'] if 'round_start' not in used_labels else None
            plt.axvline(x=marker['seconds'], color=style['color'], 
                       linestyle=style['linestyle'], alpha=style['alpha'], label=label)
            if label:
                used_labels.add('round_start')
            event_counts['round_start'] += 1
            
        # Check for stimulus
        elif marker['event_type'] == 'stimulus':
            style = marker_styles['stimulus']
            label = style['label'] if 'stimulus' not in used_labels else None
            plt.axvline(x=marker['seconds'], color=style['color'], 
                       linestyle=style['linestyle'], alpha=style['alpha'], label=label)
            if label:
                used_labels.add('stimulus')
            event_counts['stimulus'] += 1
            # Add stimulus text annotation
            if pd.notna(marker['stimulus']):
                plt.text(marker['seconds'], text_top, f"Stim: {marker['stimulus']}", 
                        rotation=90, verticalalignment='bottom')
        
        # Check for response (round column is 'response' and stimulus is 'A' or 'L')
        elif str(marker['round']).lower() == 'response' and pd.notna(marker['stimulus']):
            response = str(marker['stimulus']).upper()  # Get A/L from stimulus column
            if response in marker_styles['response']:
                style = marker_styles['response'][response]
                label = style['label'] if f'response_{response}' not in used_labels else None
                plt.axvline(x=marker['seconds'], color=style['color'], 
                           linestyle=style['linestyle'], alpha=style['alpha'], label=label)
                if label:
                    used_labels.add(f'response_{response}')
                event_counts[f'response_{response}'] += 1
                # Add response details
                if pd.notna(marker['correct']):
                    correct = str(marker['response']).lower() == 'true'  # Use response column for correctness
                    plt.text(marker['seconds'], text_bottom, 
                            f"{response} ({'✓' if correct else '✗'})", 
                            rotation=90, verticalalignment='top')
    
    print(f"\nEvents plotted:")
    print(f"Round starts: {event_counts['round_start']}")
    print(f"Stimuli shown: {event_counts['stimulus']}")
    print(f"A responses: {event_counts['response_A']}")
    print(f"L responses: {event_counts['response_L']}")
    
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

def main():
    try:
        eeg_data, timestamps, markers_df = load_latest_data()

        print(f"eeg data shape: {eeg_data.shape}")
        print(eeg_data[0])

        print(markers_df)
        
        plot_eeg_with_markers(eeg_data, timestamps, markers_df, channel=0)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 