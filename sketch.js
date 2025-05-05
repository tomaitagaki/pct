// Wait for both DOM and p5.js to be ready
let sketch = function(p) {
    // Game state variables - scoped to p5 instance
    let stimuli = ["triangle", "square", "circle"]; 
    let associations = {"triangle": 0.8, "square": 0.7, "circle": 0.6}; // probabilities
    let userResponses = [];
    let score = 0, totalTrials = 0;
    let pastTrials = 0;
    let currentStimulus = "";
    let feedback = "";
    let trialData = [];
    let roundSize = 10; // n_trials per round
    let showRoundSummary = false; 
    let currentRound = 0; 
    let inputEnabled = false;  // Start with input disabled
    let gameStarted = false;
    let wsConnected = false;
    let ws = null;
    let reconnectAttempts = 0;
    let maxReconnectAttempts = 5;
    let blockAccuracies = []; // Array to store accuracy for each block
    let blockCorrect = 0;     // Correct responses in current block
    let blockTotal = 0;       // Total responses in current block

    // DOM elements - scoped to p5 instance
    let canvas;
    let instructionScreen;
    let gameContainer;
    let startButton;
    let restartButton;

    // --- Probabilistic Categorization Task (PCT) Logic ---

    const STIMULI = ['circle', 'triangle', 'square'];
    const PROBS = [0.8, 0.7, 0.6];
    const RESPONSES = ['A', 'L'];
    const N_BLOCKS = 8;
    const SAMPLES_PER_BLOCK = 10;

    let pctAssociations = {}; // {stimulus: {prob, high, low}}
    let originalAssociation = {};
    let trialAssociations = [];
    let trialLog = [];
    let currentTrial = 0; // block index (0-7)
    let sampleInBlock = 0; // sample index within block (0-9)

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    function setupProbabilisticAssociations() {
        let shuffledProbs = PROBS.slice();
        shuffleArray(shuffledProbs);
        let associations = {};
        for (let i = 0; i < STIMULI.length; i++) {
            let stim = STIMULI[i];
            let highResp = RESPONSES[Math.floor(Math.random() * 2)];
            let lowResp = highResp === 'A' ? 'L' : 'A';
            associations[stim] = {
                prob: shuffledProbs[i],
                high: highResp,
                low: lowResp
            };
        }
        // Log each association
        for (let stim of STIMULI) {
            const a = associations[stim];
            console.log(
                `Stimulus: ${stim} | Probability: ${a.prob} | High: ${a.high} | Low: ${a.low}`
            );
        }
        return associations;
    }

    function getProbabilisticCorrectResponse(stimulus, associations) {
        let assoc = associations[stimulus];
        if (Math.random() < assoc.prob) {
            return assoc.high;
        } else {
            return assoc.low;
        }
    }

    // Setup all block associations at the start
    function setupTrials() {
        pctAssociations = setupProbabilisticAssociations();
        console.log("PCT associations:", pctAssociations);
        trialAssociations = [];
        trialLog = [];
        currentTrial = 0;
        sampleInBlock = 0;
        blockAccuracies = [];
        blockCorrect = 0;
        blockTotal = 0;

        // For blocks 1-4, 7, 8: use pctAssociations
        // For block 5: random 50/50 association (deterministic)
        // For block 6: same as block 5, but always incorrect
        for (let i = 0; i < N_BLOCKS; i++) {
            if (i === 4 || i === 5) {
                // Block 5 and 6: 50/50 random
                let assoc5050 = {};
                for (let stim of STIMULI) {
                    let highResp = RESPONSES[Math.floor(Math.random() * 2)];
                    let lowResp = highResp === 'A' ? 'L' : 'A';
                    assoc5050[stim] = {prob: 0.5, high: highResp, low: lowResp};
                }
                trialAssociations.push(assoc5050);
            } else {
                // All other blocks: use pctAssociations
                trialAssociations.push(JSON.parse(JSON.stringify(pctAssociations)));
            }
        }
    }

    function getCurrentTrialAssociation() {
        if (currentTrial >= trialAssociations.length) {
            return null;
        }
        return trialAssociations[currentTrial];
    }

    function logTrialAssociation() {
        trialLog.push({
            block: currentTrial + 1,
            association: {...trialAssociations[currentTrial]}
        });
    }

    function saveAssociationLog() {
        const logData = JSON.stringify(trialLog, null, 2);
        const blob = new Blob([logData], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'trial_associations_' + new Date().toISOString().replace(/[-:.]/g, '') + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // --- End Probabilistic Categorization Task (PCT) Logic ---

    p.preload = function() {
        // This runs first, before setup()
        console.log('p5.js preload started');
    };

    p.setup = function() {
        console.log('p5.js setup started');
        
        // Wait a short moment to ensure DOM is ready
        setTimeout(() => {
            initializeGame();
        }, 100);
    };

    function initializeGame() {
        console.log('Initializing game...');
        
        // Get DOM elements
        instructionScreen = document.getElementById('instructionScreen');
        gameContainer = document.getElementById('gameContainer');
        startButton = document.getElementById('startButton');
        restartButton = document.getElementById('restartButton');
        
        // Debug DOM elements
        console.log('DOM Elements found:', {
            instructionScreen: !!instructionScreen,
            gameContainer: !!gameContainer,
            startButton: !!startButton,
            restartButton: !!restartButton
        });
        
        if (!instructionScreen || !gameContainer || !startButton || !restartButton) {
            console.error('Required DOM elements not found');
            return;
        }
        
        // Create canvas and put it in the sketch-holder div
        let sketchHolder = document.getElementById('sketch-holder');
        if (!sketchHolder) {
            console.error('Sketch holder not found');
            return;
        }
        
        canvas = p.createCanvas(400, 400);
        canvas.parent(sketchHolder);
        
        // Add event listeners
        startButton.addEventListener('click', startGame);
        restartButton.addEventListener('click', restartGame);
        
        // Set text properties
        p.textSize(64);
        p.textAlign(p.CENTER, p.CENTER);
        p.textFont('Arial Unicode MS');  // Use a font that supports Unicode symbols
        
        // Initialize WebSocket only once
        if (!ws) {
            initializeLSL();
        }
        
        // Initially hide game container
        gameContainer.style.display = 'none';
    }

    function drawStimulus(shape, x, y, size) {
        p.push();
        p.stroke(0);
        p.strokeWeight(2);
        p.fill(255);
        
        switch(shape) {
            case "triangle":
                // Draw equilateral triangle
                let h = size * 0.866; // height of equilateral triangle
                p.triangle(
                    x, y - h/2,           // top point
                    x - size/2, y + h/2,  // bottom left
                    x + size/2, y + h/2   // bottom right
                );
                break;
            case "square":
                // Draw square
                p.rectMode(p.CENTER);
                p.rect(x, y, size, size);
                break;
            case "circle":
                // Draw circle
                p.circle(x, y, size);
                break;
        }
        p.pop();
    }

    p.draw = function() {
        p.background(220);
        
        if (!gameStarted) {
            return;  // Don't draw anything if game hasn't started
        }
        
        if (showRoundSummary) {
            p.textSize(24);
            p.text(`Round ${currentRound} Complete!`, p.width / 2, p.height / 2 - 40);
            // Show block accuracy for the most recent block
            let lastBlockIdx = currentTrial - 1;
            if (lastBlockIdx >= 0 && blockAccuracies.length > 0) {
                let lastBlockAcc = (blockAccuracies[lastBlockIdx] * 100).toFixed(2);
                p.text(`Block ${lastBlockIdx + 1} Accuracy: ${lastBlockAcc}%`, p.width / 2, p.height / 2);
            } else {
                p.text(`Block Accuracy: N/A`, p.width / 2, p.height / 2);
            }
            p.text(`Press SPACE to continue`, p.width / 2, p.height / 2 + 40);
        } else {
            // Draw stimulus
            if (currentStimulus) {
                drawStimulus(currentStimulus, p.width/2, p.height/2, 80);
            }
            
            // Draw feedback
            p.textSize(32);
            p.fill(feedback === "Correct" ? "green" : "red");
            p.text(feedback, p.width / 2, p.height / 2 + 100);
        
            p.textSize(16);
            p.fill(0);
            p.text("Press 'A' or 'L' to respond", p.width / 2, p.height - 50);
            
            // Show WebSocket status
            p.textSize(12);
            p.fill(wsConnected ? "green" : "red");
            p.text(wsConnected ? "Connected to LSL" : "Not connected to LSL", p.width / 2, 20);
        }
    };

    p.keyPressed = function() {
        if (showRoundSummary && p.key === " ") {
            showRoundSummary = false;
            startNewRound();
            return;
        }
        if (!showRoundSummary && inputEnabled && (p.key === "a" || p.key === "l")) {
            inputEnabled = false;
            let assoc = getCurrentTrialAssociation();
            if (!assoc) {
                feedback = 'No more blocks!';
                return;
            }
            let correctResponse;
            // --- Special logic for block 6 (index 5): always incorrect ---
            if (currentTrial === 5) {
                // Use 50/50 association but always mark as incorrect
                correctResponse = getProbabilisticCorrectResponse(currentStimulus, assoc);
            } else {
                correctResponse = getProbabilisticCorrectResponse(currentStimulus, assoc);
            }
            let userResponse = p.key.toUpperCase();
            let correct;
            if (currentTrial === 5) {
                correct = false;
                feedback = "Incorrect";
            } else {
                correct = (userResponse === correctResponse);
                feedback = correct ? "Correct" : "Incorrect";
            }
            score += correct ? 1 : 0;
            totalTrials++;
            blockTotal++;
            if (correct) blockCorrect++;
            let trialTimestamp = new Date().toLocaleString('sv').replace(' ', 'T');
            let accuracy = score / totalTrials;
            let currentTrialData = {
                timestamp: trialTimestamp,
                stimulus: currentStimulus,
                response: userResponse,
                correct: correct,
                accuracy: accuracy.toFixed(3),
                round: currentRound,
                block: currentTrial + 1,
                sample: sampleInBlock + 1,
                event_type: 'response'
            };
            sendMarker(currentTrialData);
            trialData.push(currentTrialData);
            console.log(currentTrialData);
            sampleInBlock++;
            if (sampleInBlock >= SAMPLES_PER_BLOCK) {
                logTrialAssociation();
                let blockAccuracy = blockTotal > 0 ? blockCorrect / blockTotal : 0;
                blockAccuracies.push(blockAccuracy);
                blockCorrect = 0;
                blockTotal = 0;
                currentTrial++;
                sampleInBlock = 0;
            }
            setTimeout(() => {
                nextTrial();
            }, 2000);
        }
    };

    // Game control functions
    function startGame() {
        // Reset game state
        userResponses = [];
        score = 0;
        totalTrials = 0;
        pastTrials = 0;
        currentStimulus = "";
        feedback = "";
        trialData = [];
        showRoundSummary = false;
        currentRound = 0;
        blockAccuracies = [];
        blockCorrect = 0;
        blockTotal = 0;
        // associations = {"triangle": 0.8, "square": 0.7, "circle": 0.6};  // Remove old probability logic

        // --- Use new trial logic ---
        setupTrials();
        currentTrial = 0;
        sampleInBlock = 0;
        // --- End new trial logic ---

        // Update UI
        gameStarted = true;
        inputEnabled = true;
        instructionScreen.style.display = 'none';
        gameContainer.style.display = 'flex';
        
        // Send game start marker with local timestamp
        sendMarker({
            timestamp: new Date().toLocaleString('sv').replace(' ', 'T'),
            event_type: 'game_start',
            round: 0
        });
        
        // Start first round
        startNewRound();
    }

    function restartGame() {
        // Send game end marker with local timestamp
        sendMarker({
            timestamp: new Date().toLocaleString('sv').replace(' ', 'T'),
            event_type: 'game_end',
            round: currentRound
        });
        // --- Save association log at end of experiment ---
        saveAssociationLog();
        // --- End save association log ---
        // Show instruction screen and hide game
        instructionScreen.style.display = 'block';
        gameContainer.style.display = 'none';
        gameStarted = false;
        inputEnabled = false;
    }

    function startNewRound() {
        currentRound++;
        // Send round start marker with local timestamp
        sendMarker({
            timestamp: new Date().toLocaleString('sv').replace(' ', 'T'),
            event_type: 'round_start',
            round: currentRound
        });
        nextTrial();
    }

    function nextTrial() {
        // Prevent running past the last block
        if (currentTrial >= trialAssociations.length) {
            feedback = 'Experiment complete!';
            inputEnabled = false;
            return;
        }
        if (totalTrials > 0 && totalTrials % roundSize === 0 && pastTrials != totalTrials) {
            showRoundSummary = true;
            pastTrials += roundSize;
            return;
        }
        // Select random stimulus from the stimuli array directly
        currentStimulus = stimuli[Math.floor(p.random(stimuli.length))];
        console.log('Selected stimulus:', currentStimulus);  // Debug log
        // Send stimulus marker with local timestamp
        sendMarker({
            timestamp: new Date().toLocaleString('sv').replace(' ', 'T'),
            event_type: 'stimulus',
            stimulus: currentStimulus,
            round: currentRound,
            block: currentTrial + 1,
            sample: sampleInBlock + 1
        });
        feedback = "";
        inputEnabled = true; 
    }

    // WebSocket functions
    function initializeLSL() {
        if (reconnectAttempts >= maxReconnectAttempts) {
            console.log("Max reconnection attempts reached. Please refresh the page to try again.");
            return;
        }

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
                reconnectAttempts = 0;  // Reset attempts on successful connection
                // Send a test message
                sendMarker("TEST");
            };
            
            ws.onclose = (event) => {
                console.log("WebSocket connection closed", event.code, event.reason);
                wsConnected = false;
                reconnectAttempts++;
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    // Try to reconnect after 2 seconds
                    setTimeout(initializeLSL, 2000);
                } else {
                    console.log("Max reconnection attempts reached. Please refresh the page to try again.");
                }
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
            reconnectAttempts++;
            if (reconnectAttempts < maxReconnectAttempts) {
                // Try to reconnect after 2 seconds
                setTimeout(initializeLSL, 2000);
            }
        }
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
};

// Only create p5 instance after DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM Content Loaded, creating p5 instance');
        new p5(sketch);
    });
} else {
    // DOM is already ready
    console.log('DOM already ready, creating p5 instance');
    new p5(sketch);
}
