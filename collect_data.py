import numpy as np
import pylsl
import time
from datetime import datetime
import os
import pandas as pd
import json
import websockets
import asyncio
import argparse

async def send_eeg_start_marker(eeg_start_time):
    """Send EEG start time to marker server"""
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            marker_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'eeg_start',
                'eeg_start_time': eeg_start_time
            }
            await websocket.send(json.dumps(marker_data))
            print("Sent EEG start marker to server")
    except Exception as e:
        print(f"Warning: Could not send EEG start marker: {e}")

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

def collect_data(duration=None, file_name=None):
    """Collect EEG data for specified duration
    
    Args:
        duration: Recording duration in seconds
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
    
    openbci_stream, _ = find_lsl_streams()  # Still check for marker stream to ensure system is ready
    
    openbci_inlet = pylsl.StreamInlet(
        openbci_stream,
        max_buflen=360,  
        max_chunklen=12
    )
    
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
        return
    else:
        print(f"Successfully received test sample: shape={len(test_sample)}")
    
    # Send EEG start marker
    asyncio.run(send_eeg_start_marker(time.time()))
    
    last_sample_time = time.time()
    sample_timeout_count = 0
    
    try:
        total_samples = 0
        last_print_time = time.time()
        print_interval = 5.0 
        
        while time.time() - last_sample_time < duration if duration else True:
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
            
            # Print stats periodically
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - last_sample_time
                sample_rate = total_samples/elapsed if elapsed > 0 else 0
                print(f"\rRecording: {int(elapsed)}s | "
                      f"Samples: {total_samples} ({sample_rate:.1f}/s | "
                      f"Expected: {openbci_info.nominal_srate():.1f}/s)", end='')
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
        
        # Save EEG data
        np.save(os.path.join(save_dir, f'eeg_data_{base_name}.npy'), eeg_data)
        np.save(os.path.join(save_dir, f'timestamps_{base_name}.npy'), eeg_timestamps)
        
        print(f"\nData saved:")
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Files saved with base name: {base_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=None, help='Recording duration in seconds (default: infinite)')
    parser.add_argument('--file_name', type=str, default=None, help='Custom file name for output files (no extension)')
    args = parser.parse_args()
    
    collect_data(duration=args.duration, file_name=args.file_name) 