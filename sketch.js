let stimuli = ["▲", "■", "●"]; 
let associations = {"▲": 0.8, "■": 0.7, "●": 0.6}; // probabilities
let userResponses = [];
let score = 0, totalTrials = 0;
let pastTrials = 0;
let currentStimulus = "";
let feedback = "";
let trialData = [];
let roundSize = 10; // n_trials per round
let showRoundSummary = false; 
let currentRound = 0; 
let inputEnabled = true; 
let ws = null;
let wsConnected = false;

// try lsl via websocket
function initializeLSL() {
    console.log("Attempting to connect to WebSocket...");
    try {
        if (ws) {
            console.log("Closing existing WebSocket connection");
            ws.close();
        }
        
        ws = new WebSocket('ws://localhost:8765');
        
        ws.onopen = () => {
            console.log("WebSocket connection established");
            wsConnected = true;
            // Send a test message
            sendMarker("TEST");
        };
        
        ws.onclose = (event) => {
            console.log("WebSocket connection closed", event.code, event.reason);
            wsConnected = false;
            // Try to reconnect after 2 seconds
            setTimeout(initializeLSL, 2000);
        };
        
        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            wsConnected = false;
        };

        ws.onmessage = (event) => {
            console.log("Received message from server:", event.data);
        };
    } catch (error) {
        console.error("Error creating WebSocket:", error);
        wsConnected = false;
        // Try to reconnect after 2 seconds
        setTimeout(initializeLSL, 2000);
    }
}

function setup() {
    createCanvas(400, 400);
    textSize(64);
    textAlign(CENTER, CENTER);
    initializeLSL();
    startNewRound();
}

function draw() {
    background(220);
    
    if (showRoundSummary) {
        textSize(24);
        text(`Round ${currentRound} Complete!`, width / 2, height / 2 - 40);
        text(`Accuracy: ${(score / totalTrials * 100).toFixed(2)}%`, width / 2, height / 2);
        text(`Press SPACE to continue`, width / 2, height / 2 + 40);
    } else {
        text(currentStimulus, width / 2, height / 2);
        textSize(32);
        fill(feedback === "Correct" ? "green" : "red");
        text(feedback, width / 2, height / 2 + 100);
    
        textSize(16);
        fill(0);
        text("Press 'A' or 'L' to respond", width / 2, height - 50);
        
        // Show WebSocket status
        textSize(12);
        fill(wsConnected ? "green" : "red");
        text(wsConnected ? "Connected to LSL" : "Not connected to LSL", width / 2, 20);
    }
}

function startNewRound() {
    currentRound++;
    // Send round start marker
    sendMarker({
        timestamp: new Date().toISOString(),
        event_type: 'round_start',
        round: currentRound
    });
    nextTrial();
}

function nextTrial() {
    if (totalTrials > 0 && totalTrials % roundSize === 0 && pastTrials != totalTrials) {
        showRoundSummary = true;
        pastTrials += roundSize;
        return;
    }

    let keys = Object.keys(associations);
    currentStimulus = random(keys);
    
    // Send stimulus marker
    sendMarker({
        timestamp: new Date().toISOString(),
        event_type: 'stimulus',
        stimulus: currentStimulus,
        round: currentRound
    });
    
    feedback = "";
    inputEnabled = true; 
}

function sendMarker(markerData) {
    if (!ws) {
        console.log("No WebSocket connection exists, attempting to connect...");
        initializeLSL();
        return;
    }
    
    if (ws.readyState === WebSocket.OPEN) {
        const message = JSON.stringify(markerData);
        console.log("Sending marker:", message);
        ws.send(message);
    } else {
        console.log("WebSocket not ready:", ws.readyState);
        if (ws.readyState === WebSocket.CLOSED) {
            console.log("WebSocket is closed, attempting to reconnect...");
            initializeLSL();
        }
    }
}

function keyPressed() {
    if (showRoundSummary && key === " ") {
        showRoundSummary = false;
        startNewRound();
        return;
    }

    if (!showRoundSummary && inputEnabled && (key === "a" || key === "l")) {
        inputEnabled = false;
        let correct = random() < associations[currentStimulus];
        feedback = correct ? "Correct" : "Incorrect";
        score += correct ? 1 : 0;
        totalTrials++;

        let trialTimestamp = new Date().toISOString();
        let accuracy = score / totalTrials;
        
        // Create complete trial data
        let currentTrial = {
            timestamp: trialTimestamp,
            stimulus: currentStimulus,
            response: key.toUpperCase(),
            correct: correct,
            accuracy: accuracy.toFixed(3),
            round: currentRound,
            event_type: 'response'  // Added for the plotting script
        };
        
        // Send complete trial data to marker server
        sendMarker(currentTrial);

        // Add to trial history
        trialData.push(currentTrial);

        console.log(currentTrial);

        if (totalTrials % roundSize === 0) {
            saveData(); 
        }

        adaptDifficulty();
        
        setTimeout(() => {
            nextTrial();
        }, 2000); 
    }
}

function adaptDifficulty() {
    let accuracy = score / totalTrials;
    if (accuracy > 0.8) {
        for (let key in associations) {
            associations[key] = max(associations[key] - 0.05, 0.5);
        }
    }
    if (accuracy < 0.6) {
        for (let key in associations) {
            associations[key] = min(associations[key] + 0.05, 0.9);
        }
    }
}

function saveData() {
    let csvContent = "timestamp,stimulus,response,correct,accuracy,round\n";
    trialData.forEach(trial => {
        csvContent += `${trial.timestamp},${trial.stimulus},${trial.response},${trial.correct},${trial.accuracy},${trial.round}\n`;
    });

    // Use the Fetch API to send the data to the server
    fetch('/save-data', {
        method: 'POST',
        headers: {
            'Content-Type': 'text/csv'
        },
        body: csvContent
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        console.log('Data saved successfully');
    })
    .catch(error => {
        console.error('Error saving data:', error);
        let blob = new Blob([csvContent], { type: "text/csv" });
        let a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "PCT_trial_data.csv";
        a.click();
    });
}
