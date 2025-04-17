import pylsl
import asyncio
import websockets
import json
from datetime import datetime
import pandas as pd
import os
import csv

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

def save_marker_to_csv(data, csv_path):
    """Save marker data to CSV file"""
    # Skip test messages
    if data == "TEST":
        return
        
    try:
        # Ensure all marker types have consistent columns
        marker_row = {
            'timestamp': data.get('timestamp', ''),
            'event_type': data.get('event_type', ''),
            'stimulus': data.get('stimulus', ''),
            'response': data.get('response', ''),
            'correct': data.get('correct', ''),
            'accuracy': data.get('accuracy', ''),
            'round': data.get('round', '')
        }
        
        # Convert data to DataFrame row
        df_row = pd.DataFrame([marker_row])
        
        # If file exists, append without header. If not, create with header
        if os.path.exists(csv_path):
            df_row.to_csv(csv_path, mode='a', header=False, index=False, quoting=csv.QUOTE_MINIMAL)
        else:
            df_row.to_csv(csv_path, mode='w', header=True, index=False, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        print(f"Error saving marker to CSV: {e}")
        print(f"Data was: {data}")

async def handle_websocket(websocket, marker_outlet):
    """Handle incoming websocket messages and send them to LSL"""
    print("\n=== New PCT client connected ===")
    
    # Create a new CSV file for this session with proper headers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join('data', f'markers_{timestamp}.csv')
    os.makedirs('data', exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_path):
        headers = ['timestamp', 'event_type', 'stimulus', 'response', 'correct', 'accuracy', 'round']
        pd.DataFrame(columns=headers).to_csv(csv_path, index=False)
    
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
                save_marker_to_csv(marker_data, csv_path)
                print(f"✓ Complete marker data saved to CSV")
                
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON message received: {message}")
            except Exception as e:
                print(f"Error processing message: {e}")
                print(f"Message was: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("\n=== PCT client disconnected ===")

async def main():
    """Start the marker server"""
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n=== Marker server stopped ===") 