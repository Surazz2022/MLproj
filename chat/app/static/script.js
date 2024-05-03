document.addEventListener('DOMContentLoaded', function() {
    const chatLog = document.getElementById('chat-log');
    const form = document.getElementById('chat-form');

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const userInput = document.getElementById('input').value;
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input: userInput })
        })
        .then(response => response.json())
        .then(data => {
            const response = data.response;
            const chatLogEntry = document.createElement('div');
            chatLogEntry.textContent = `You: ${userInput}\nBot: ${response}`;
            chatLog.appendChild(chatLogEntry);
            document.getElementById('input').value = '';
        })
        .catch(error => console.error(error));
    });
});