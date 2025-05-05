import numpy as np
import pylsl
import time
from datetime import datetime
import os
import pandas as pd
import threading
from queue import Queue
import argparse

def find_openbci_stream():
    """Find OpenBCI LSL stream"""
    print("Looking for OpenBCI LSL stream...")
    
    # Find OpenBCI stream (wait up to 10 seconds)
    print("\nLooking for any available streams...")
    all_streams = pylsl.resolve_streams(wait_time=10.0)
    print(f"Found {len(all_streams)} total streams")
    
    # Find OpenBCI stream
    openbci_streams = []
    for stream in all_streams:
        if stream.type() == 'EEG' or 'Time Series' in stream.name():
            openbci_streams.append(stream)
    
    if not openbci_streams:
        raise RuntimeError(
            "No OpenBCI LSL stream found. Please check:\n"
            "1. OpenBCI GUI is running\n"
            "2. In the Networking widget:\n"
            "   - Protocol is set to 'LSL' (not UDP)\n"
            "   - Stream 1 Data Type is set to 'Time Series'\n"
            "   - 'Start LSL Stream' button has been clicked"
        )
    
    openbci_stream = openbci_streams[0]
    print(f"\nFound OpenBCI stream:")
    print(f"- Name: {openbci_stream.name()}")
    print(f"- Type: {openbci_stream.type()}")
    print(f"- Source ID: {openbci_stream.source_id()}")
    print(f"- Channel count: {openbci_stream.channel_count()}")
    print(f"- Sampling rate: {openbci_stream.nominal_srate()} Hz")
    print(f"- Format: {openbci_stream.channel_format()}")
    
    return openbci_stream

def save_marker(marker_value, csv_path):
    """Save marker to CSV file"""
    try:
        # Initialize marker row
        marker_row = {
            'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            'marker': marker_value
        }
        
        # Convert to DataFrame row
        df_row = pd.DataFrame([marker_row])
        
        # If file exists, append without header. If not, create with header
        if os.path.exists(csv_path):
            df_row.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_path, mode='w', header=True, index=False)
            
        print(f"✓ Marker: {marker_value}")
    except Exception as e:
        print(f"Error saving marker: {e}")

def collect_data(marker_queue, duration=None, file_name=None):
    """Collect OpenBCI data and handle markers
    
    Args:
        marker_queue: Queue for receiving markers from input thread
        duration: Recording duration in seconds (None for infinite)
        file_name: Custom file name for output files (no extension)
    """
    # Create data directory
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for this session
    if file_name:
        base_name = file_name
    else:
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize files
    eeg_data = []
    eeg_timestamps = []
    csv_path = os.path.join(save_dir, f'markers_{base_name}.csv')
    
    # Set up OpenBCI stream
    openbci_stream = find_openbci_stream()
    openbci_inlet = pylsl.StreamInlet(
        openbci_stream,
        max_buflen=360,
        max_chunklen=12
    )
    
    print("\nTesting OpenBCI stream...")
    test_sample, test_timestamp = openbci_inlet.pull_sample(timeout=5.0)
    if test_sample is None:
        print("WARNING: Could not get initial sample from OpenBCI. Check if data is being sent.")
        return
    print(f"Successfully received test sample: shape={len(test_sample)}")
    
    print("\nRecording started. Available commands:")
    print("Enter a number (0-9) to add a marker")
    print("q - Quit recording")
    
    start_time = time.time()
    last_sample_time = time.time()
    last_print_time = time.time()
    print_interval = 5.0
    total_samples = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Check duration
            if duration and current_time - start_time >= duration:
                print(f"\nRecording duration ({duration}s) reached")
                break
            
            # Get EEG data
            samples, timestamps = openbci_inlet.pull_chunk(timeout=0.0)
            if samples:
                eeg_data.extend(samples)
                eeg_timestamps.extend(timestamps)
                total_samples += len(samples)
                last_sample_time = current_time
            
            # Check for markers in queue
            while not marker_queue.empty():
                marker = marker_queue.get().strip()
                if marker.lower() == 'q':
                    print("\nQuit command received")
                    raise KeyboardInterrupt
                
                # Try to parse marker as number
                try:
                    marker_value = int(marker)
                    save_marker(marker_value, csv_path)
                except ValueError:
                    # Ignore invalid input
                    pass
            
            # Print stats periodically
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - start_time
                sample_rate = total_samples/elapsed if elapsed > 0 else 0
                print(f"\rRecording: {int(elapsed)}s | "
                      f"Samples: {total_samples} ({sample_rate:.1f} Hz)", end='')
                last_print_time = current_time
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        # Save EEG data
        if eeg_data:
            eeg_data = np.array(eeg_data)
            eeg_timestamps = np.array(eeg_timestamps)
            np.save(os.path.join(save_dir, f'eeg_data_{base_name}.npy'), eeg_data)
            np.save(os.path.join(save_dir, f'timestamps_{base_name}.npy'), eeg_timestamps)
            print(f"\nData saved:")
            print(f"EEG data shape: {eeg_data.shape}")
            print(f"Files saved with base name: {base_name}")

def input_thread(marker_queue):
    """Thread for handling terminal input"""
    while True:
        try:
            command = input()
            marker_queue.put(command)
            if command.lower() == 'q':
                break
        except EOFError:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=None, help='Recording duration in seconds (default: infinite)')
    parser.add_argument('--file_name', type=str, default=None, help='Custom file name for output files (no extension)')
    args = parser.parse_args()
    
    # Create queue for markers
    marker_queue = Queue()
    
    # Start input thread
    input_thread = threading.Thread(target=input_thread, args=(marker_queue,))
    input_thread.daemon = True
    input_thread.start()
    
    # Start data collection
    collect_data(marker_queue, duration=args.duration, file_name=args.file_name) 