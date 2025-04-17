import pylsl
import time

print("Looking for all LSL streams...")
streams = pylsl.resolve_streams()

if not streams:
    print("No LSL streams found!")
else:
    print(f"\nFound {len(streams)} streams:")
    for i, stream in enumerate(streams):
        print(f"\nStream {i+1}:")
        print(f"- Name: {stream.name()}")
        print(f"- Type: {stream.type()}")
        print(f"- Channel count: {stream.channel_count()}")
        print(f"- Sampling rate: {stream.nominal_srate()} Hz")
        print(f"- Source ID: {stream.source_id()}")
        print(f"- Format: {stream.channel_format()}")
        
        # If this is the OpenBCI stream, try to get some data
        if stream.type() == 'EEG' and stream.name() == 'obci_eeg1':
            print("\nTesting OpenBCI data stream...")
            inlet = pylsl.StreamInlet(stream)
            
            # Try to get data for 5 seconds
            start_time = time.time()
            samples_received = 0
            
            while time.time() - start_time < 5:
                sample, timestamp = inlet.pull_sample(timeout=0.1)
                if sample:
                    samples_received += 1
                    if samples_received == 1:
                        print(f"First sample received! Values: {sample}")
                    
            print(f"Received {samples_received} samples in 5 seconds")
            print(f"Effective sampling rate: {samples_received/5:.1f} Hz")
            print(f"Expected sampling rate: {stream.nominal_srate()} Hz")
            
            if samples_received == 0:
                print("\nNo data received! Please check:")
                print("1. Is the main OpenBCI GUI data stream running?")
                print("2. Did you click 'Start LSL Stream' in the Networking widget?")
                print("3. Are you seeing data in the Time Series widget?")

print("\nPress Ctrl+C to exit") 