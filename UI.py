<!-- Example Chat UI -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Chat</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f4f4; }
        #chat-container { width: 400px; margin: 40px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 20px; }
        #messages { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; background: #fafafa; }
        .message { margin: 8px 0; }
        .user { color: #0078d7; }
        .ai { color: #d17a00; }
        #input-area { display: flex; }
        #user-input { flex: 1; padding: 8px; }
        #send-btn { padding: 8px 16px; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="user-input" placeholder="Type your message..." autocomplete="off" />
            <button id="send-btn">Send</button>
        </div>
    </div>
    <script>
        const messages = document.getElementById('messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');

        function appendMessage(sender, text) {
            const msgDiv = document.createElement('div');
            msgDiv.className = 'message ' + sender;
            msgDiv.textContent = (sender === 'user' ? 'You: ' : 'AI: ') + text;
            messages.appendChild(msgDiv);
            messages.scrollTop = messages.scrollHeight;
