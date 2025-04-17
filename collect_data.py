import numpy as np
import pylsl
import time
from datetime import datetime
import os
import pandas as pd

def find_lsl_streams():
    """Find available LSL streams for OpenBCI and PCT"""
    print("Looking for LSL streams...")
    
    # Wait for streams to become available
    print("Waiting for LSL streams to become available...")
    print("Note: Make sure you're using LSL protocol in the OpenBCI GUI, not UDP!")
    
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
    
    # Find PCT stream (wait up to 10 seconds)
    pct_streams = pylsl.resolve_byprop('type', 'Markers', timeout=10.0)
    if not pct_streams:
        raise RuntimeError("No PCT marker stream found. Make sure the PCT application is running.")
    pct_stream = pct_streams[0]
    print(f"\nFound PCT stream:")
    print(f"- Name: {pct_stream.name()}")
    print(f"- Type: {pct_stream.type()}")
    print(f"- Source ID: {pct_stream.source_id()}")
    
    return openbci_stream, pct_stream

def collect_data(duration=300, save_dir='data'):
    """Collect data from both streams for specified duration
    
    Args:
        duration: Recording duration in seconds
        save_dir: Directory to save the data
    """
    os.makedirs(save_dir, exist_ok=True)
    
    openbci_stream, pct_stream = find_lsl_streams()
    
    openbci_inlet = pylsl.StreamInlet(
        openbci_stream,
        max_buflen=360,  
        max_chunklen=12
    )
    pct_inlet = pylsl.StreamInlet(pct_stream)
    
    openbci_info = openbci_inlet.info()
    print("\nOpenBCI Stream Info:")
    print(f"- Name: {openbci_info.name()}")
    print(f"- Type: {openbci_info.type()}")
    print(f"- Channel count: {openbci_info.channel_count()}")
    print(f"- Sampling rate: {openbci_info.nominal_srate()} Hz")
    print(f"- Channel format: {openbci_info.channel_format()}")
    
    print("\nTesting OpenBCI stream...")
    test_sample, test_timestamp = openbci_inlet.pull_sample(timeout=5.0)
    if test_sample is None:
        print("WARNING: Could not get initial sample from OpenBCI. Check if data is being sent.")
    else:
        print(f"Successfully received test sample: shape={len(test_sample)}")
    
    eeg_data = []
    eeg_timestamps = []
    if test_sample is not None:
        eeg_data.append(test_sample)
        eeg_timestamps.append(test_timestamp)
    
    markers = []
    marker_timestamps = []
    
    print(f"\nRecording for {duration} seconds...")
    start_time = time.time()
    last_sample_time = time.time()
    sample_timeout_count = 0
    
    try:
        total_samples = len(eeg_data)
        total_markers = 0
        last_print_time = time.time()
        print_interval = 5.0 
        
        while time.time() - start_time < duration:
            samples, timestamps = openbci_inlet.pull_chunk(timeout=0.0)
            current_time = time.time()
            
            if samples:
                eeg_data.extend(samples)
                eeg_timestamps.extend(timestamps)
                total_samples += len(samples)
                last_sample_time = current_time
                sample_timeout_count = 0
            else:
                if current_time - last_sample_time > 1.0:  # No samples for 1 second
                    sample_timeout_count += 1
                    if sample_timeout_count == 1:  # Only print once
                        print("\nWARNING: No EEG samples received for 1 second. Check OpenBCI stream.")
            
            # Get markers
            marker, marker_time = pct_inlet.pull_sample(timeout=0.0)
            if marker:
                markers.append(marker[0])
                marker_timestamps.append(marker_time)
                total_markers += 1
                print(f"\nNew marker received: {marker[0]} at time {marker_time:.3f}")
            
            # Print stats periodically
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - start_time
                sample_rate = total_samples/elapsed if elapsed > 0 else 0
                print(f"\rRecording: {int(elapsed)}/{duration}s | "
                      f"Samples: {total_samples} ({sample_rate:.1f}/s | "
                      f"Expected: {openbci_info.nominal_srate():.1f}/s) | "
                      f"Markers: {total_markers}", end='')
                last_print_time = current_time
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        if not eeg_data:
            print("\nERROR: No EEG data was collected!")
            return
            
        # Convert to numpy arrays
        eeg_data = np.array(eeg_data)
        eeg_timestamps = np.array(eeg_timestamps)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save EEG data
        np.save(os.path.join(save_dir, f'eeg_data_{timestamp}.npy'), eeg_data)
        np.save(os.path.join(save_dir, f'timestamps_{timestamp}.npy'), eeg_timestamps)
        
        # Save markers
        marker_df = pd.DataFrame({
            'timestamp': marker_timestamps,
            'response': markers
        })
        marker_df.to_csv(os.path.join(save_dir, f'markers_{timestamp}.csv'), index=False)
        
        print(f"\nData saved:")
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Number of markers: {len(markers)}")
        print(f"Files saved with timestamp: {timestamp}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=300, help='Recording duration in seconds')
    parser.add_argument('--save-dir', type=str, default='data', help='Directory to save data')
    args = parser.parse_args()
    
    collect_data(duration=args.duration, save_dir=args.save_dir) 