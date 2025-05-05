import pylsl
import asyncio
import websockets
import json
from datetime import datetime
import pandas as pd
import os
import csv
import argparse

# Global variable to store the CSV path
MARKER_CSV_PATH = None

def create_marker_stream():
    info = pylsl.StreamInfo(
        name='PCT_Markers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id='pct_marker_stream'
    )
    return pylsl.StreamOutlet(info)

def save_marker_to_csv(data):
    """Save marker data to CSV file"""
    global MARKER_CSV_PATH
    
    # Skip test messages
    if data == "TEST":
        return
        
    try:
        # Initialize empty marker row with all columns
        marker_row = {
            'timestamp': '',
            'event_type': '',
            'eeg_start_time': '',
            'stimulus': '',
            'response': '',
            'correct': '',
            'accuracy': '',
            'round': ''
        }
        
        # Update with common fields
        marker_row['timestamp'] = data.get('timestamp', '')
        marker_row['event_type'] = data.get('event_type', '')
        
        # Handle different event types
        if data.get('event_type') == 'eeg_start':
            marker_row['eeg_start_time'] = data.get('eeg_start_time', '')
        elif data.get('event_type') == 'stimulus':
            marker_row['stimulus'] = data.get('stimulus', '')
            marker_row['round'] = data.get('round', '')
        elif data.get('event_type') == 'response':
            marker_row['stimulus'] = data.get('stimulus', '')  # The stimulus being responded to
            marker_row['response'] = data.get('response', '')
            marker_row['correct'] = data.get('correct', '')
            marker_row['accuracy'] = data.get('accuracy', '')
            marker_row['round'] = data.get('round', '')
        elif data.get('event_type') in ['round_start', 'game_start', 'game_end']:
            marker_row['round'] = data.get('round', '')
        
        # Convert data to DataFrame row
        df_row = pd.DataFrame([marker_row])
        
        # If file exists, append without header. If not, create with header
        if os.path.exists(MARKER_CSV_PATH):
            df_row.to_csv(MARKER_CSV_PATH, mode='a', header=False, index=False, quoting=csv.QUOTE_MINIMAL)
        else:
            df_row.to_csv(MARKER_CSV_PATH, mode='w', header=True, index=False, quoting=csv.QUOTE_MINIMAL)
            
        print(f"✓ Marker saved to CSV: {marker_row}")
    except Exception as e:
        print(f"Error saving marker to CSV: {e}")
        print(f"Data was: {data}")

async def handle_websocket(websocket, marker_outlet):
    """Handle incoming websocket messages and send them to LSL"""
    print("\n=== New PCT client connected ===")
    
    try:
        async for message in websocket:
            try:
                # Handle TEST message
                if message == "TEST":
                    print("\nReceived test message")
                    marker_outlet.push_sample(["TEST"])
                    print("✓ Test marker sent to LSL")
                    continue
                
                # Parse the complete marker data
                marker_data = json.loads(message)
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"\n[{current_time}] Received marker data: {marker_data}")
                
                # Send to LSL (still just sending response for compatibility)
                if 'response' in marker_data:
                    marker_outlet.push_sample([marker_data['response']])
                    print(f"✓ Response marker '{marker_data['response']}' sent to LSL")
                
                # Save complete marker data to CSV
                save_marker_to_csv(marker_data)
                
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON message received: {message}")
            except Exception as e:
                print(f"Error processing message: {e}")
                print(f"Message was: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("\n=== PCT client disconnected ===")

def main():
    global MARKER_CSV_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default=None, help='Custom file name for marker file (no extension)')
    args = parser.parse_args()
    if args.file_name:
        base_name = args.file_name
    else:
        base_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    MARKER_CSV_PATH = os.path.join('data', f'markers_{base_name}.csv')
    # Create the marker file with headers if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(MARKER_CSV_PATH):
        with open(MARKER_CSV_PATH, 'w') as f:
            f.write('timestamp,event_type,eeg_start_time,stimulus,response,correct,accuracy,round\n')
    # Start the asyncio server as before
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n=== Marker server stopped ===")

# Rename the original async main to main_async
async def main_async():
    # Create LSL marker stream
    marker_outlet = create_marker_stream()
    print("\n=== Marker Server Started ===")
    print("Created LSL marker stream 'PCT_Markers'")
    
    # Start websocket server
    async with websockets.serve(
        lambda ws: handle_websocket(ws, marker_outlet),
        "localhost",
        8765
    ):
        print("Listening for WebSocket connections on ws://localhost:8765")
        print("Waiting for PCT responses...\n")
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    main() 