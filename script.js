const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');

function appendMessage(content, sender = 'user') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    msgDiv.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

chatForm.addEventListener('submit', function(e) {
    e.preventDefault();
    const userMsg = chatInput.value.trim();
    if (!userMsg) return;
    appendMessage(userMsg, 'user');
    chatInput.value = '';

    // Simulate bot reply
    setTimeout(() => {
        appendMessage('You said: ' + userMsg, 'bot');
    }, 600);
});
Kasutaja\Documents\script.js
const loginContainer = document.getElementById('loginContainer');
const chatContainer = document.getElementById('chatContainer');
const loginForm = document.getElementById('loginForm');
const loginError = document.getElementById('loginError');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatMessages = document.getElementById('chatMessages');

function appendMessage(content, sender = 'user') {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}`;
    msgDiv.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Handle login
loginForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();

    try {
        const response = await fetch('http://localhost:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ username, password })
        });
        const data = await response.json();
        if (response.ok) {
            loginContainer.style.display = 'none';
            chatContainer.style.display = '';
        } else {
            loginError.textContent = data.error || 'Login failed';
        }
    } catch (err) {
        loginError.textContent = 'Could not connect to backend.';
    }
});

// Handle chat
chatForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const userMsg = chatInput.value.trim();
    if (!userMsg) return;
    appendMessage(userMsg, 'user');
    chatInput.value = '';

    try {
        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ message: userMsg })
        });
        const data = await response.json();
        appendMessage(data.reply, 'bot');
    } catch (err) {
        appendMessage('Error: Could not reach backend.', 'bot');
    }
});
// ...existing code...

const musicList = [
    'https://www.youtube.com/watch?v=Dc3AovUZgvo.mp3',
    'https://www.youtube.com/watch?v=lLLL1KxpYMA.mp3',
    'https://www.youtube.com/watch?v=FiThjLcB1Qk.mp3',
    'https://www.youtube.com/watch?v=yEJEO0mvxSI.mp3'
];

function playRandomMusic() {
    const audio = document.getElementById('audioPlayer');
    const randomIndex = Math.floor(Math.random() * musicList.length);
    audio.src = musicList[randomIndex];
    audio.play();
}

document.getElementById('chatForm').addEventListener('submit', function(e) {
    // ...existing chat send logic...
    playRandomMusic();
});