let ws;

function sendMessage(event) {
    event.preventDefault();

    const user_input = document.querySelector('.prompt-textarea').value.trim();
    if (user_input !== "") {
        const user_input_container = document.querySelector('.user-input-container');

        // Create container for user question and response
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container';

        // Add user question to the container
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';
        messageDiv.textContent = user_input;
        messageContainer.appendChild(messageDiv);

        // Append the message container to the output container
        const outputContainer = document.querySelector('.llms-output-container');
        outputContainer.appendChild(messageContainer);

        document.querySelector('.prompt-textarea').value = '';

        // Show user input container if it's hidden
        user_input_container.style.display = 'block';

        // Hide background-chatbot container
        document.querySelector('.background-chatbot').style.display = 'none';
        // Show conver-container instead
        document.querySelector('.conver-container').style.display = 'block';

        // Check if ws is defined before sending message
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(user_input);
        } else {
            console.error("WebSocket connection is not open or ws is undefined");
            // Handle the error appropriately
        }
    }
}

document.addEventListener("DOMContentLoaded", function() {
    ws = new WebSocket("ws://localhost:8000/ws"); // Initialize ws variable here

    ws.onopen = function() {
        console.log("WebSocket connection opened");
    };

    ws.onmessage = function(event) {
        console.log("Received message:", event.data);
        try {
            const data = JSON.parse(event.data);
            displayLLMResponse(data);
        } catch (error) {
            console.error("Error parsing JSON:", error);
        }
    };

    ws.onclose = function() {
        console.log("WebSocket connection closed");
    };

    function displayLLMResponse(data) {
        const outputContainer = document.querySelector('.llms-output-container');
    
        if (data.gpt3_answer || data.gpt4_answer) {
            // Create container for response
            const responseContainer = document.createElement('div');
            responseContainer.className = 'response-container';
    
            if (data.gpt3_answer) {
                const gpt3Div = document.createElement('div');
                gpt3Div.className = 'llm-result llm1';
                gpt3Div.textContent = 'GPT-3: ' + data.gpt3_answer;
                responseContainer.appendChild(gpt3Div);
            }
    
            if (data.gpt4_answer) {
                const gpt4Div = document.createElement('div');
                gpt4Div.className = 'llm-result llm2';
                gpt4Div.textContent = 'GPT-4: ' + data.gpt4_answer;
                responseContainer.appendChild(gpt4Div);
            }
    
            // Append the response container to the output container
            outputContainer.appendChild(responseContainer);
        }
    
        outputContainer.style.display = 'block';
    }

});
