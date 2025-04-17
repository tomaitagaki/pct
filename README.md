# Probabilistic Categorization Task (PCT)

## Steps to start data collection

1. Set up OpenBCI connection via LSL in OpenBCI GUI 
2. Start PCT web server 
3. Start the marker server to handle PCT events
4. Start data collection

## Start OpenBCI LSL

1. Open the GUI
2. Connect to the cyton board
3. In "Networking" set the mode to LSL: [link](https://docs.openbci.com/Software/CompatibleThirdPartySoftware/LSL/)
4. Ensure chaannel 1 is sending "TimeseriesRaw"
5. Start LSL
6. Start Stream
7. Leave both the stream and LSL on as you collect data

Hookup guide: [link](https://docs.openbci.com/GettingStarted/Biosensing-Setups/ExGSetup/)

## Start PCT Web Server

1. Run the following command:
   ```bash
   python -m http.server 8000
   ```
2. Open browser to http://localhost:8000
3. This should show the interactive PCT web app

## Start Marker Server

1. Run the following command:
   ```Bash
   python marker_server.py
   ```
2. This should start listening to websockets on port 8765

## Start Data Collection

1. Run the following command:
   ```Bash
   python collect_data.py
   ```
2. This will record both EEG and markers
