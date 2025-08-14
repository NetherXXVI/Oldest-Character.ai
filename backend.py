</html>
// ...existing code...
sendBtn.onclick = async () => {
    const text = userInput.value.trim();
    if (!text) return;
    appendMessage('user', text);
    userInput.value = '';
    appendMessage('ai', '...replying...');

    // Send user message to backend and get AI response
    try {
        const response = await fetch('http://localhost:5000/chat', { // Change URL to your backend
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        const data = await response.json();
        // Remove the "...replying..." message
        messages.lastChild.remove();
        appendMessage('ai', data.reply);
    } catch (error) {
        messages.lastChild.remove();
        appendMessage('ai', 'Error: Could not reach backend.');
    }
};
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    # TODO: Replace this with real AI logic
    ai_reply = f"You said: {user_message}"
    return jsonify({'reply': ai_reply})
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret in production
CORS(app, supports_credentials=True)  # Allow credentials for session cookies

# Store hashed passwords
USERS = {
    "testuser": generate_password_hash("testpassword"),
    "anotheruser": generate_password_hash("anotherpassword")
}
# In-memory chat history: {username: [ {role: 'user'/'ai', message: ...}, ... ]}
CHAT_HISTORY = {}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('username'):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username in USERS and check_password_hash(USERS[username], password):
        session['username'] = username
        # Initialize chat history if not present
        if username not in CHAT_HISTORY:
            CHAT_HISTORY[username] = []
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid credentials'}), 401
    

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('username'):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if USERS.get(username) == password:
        session['username'] = username
        # Initialize chat history if not present
        if username not in CHAT_HISTORY:
            CHAT_HISTORY[username] = []
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'message': 'Logged out'})

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    username = session['username']
    data = request.get_json()
    user_message = data.get('message', '')
    # Store user message
    CHAT_HISTORY[username].append({'role': 'user', 'message': user_message})
    # AI mimics the user
    ai_reply = f"If I were you, I'd say: {user_message}"
    # Generate AI reply (replace with real AI logic if needed)
    ai_reply = f"You said: {user_message}"
    # Store AI reply
    CHAT_HISTORY[username].append({'role': 'ai', 'message': ai_reply})
    return jsonify({'reply': ai_reply})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if USERS.get(username) == password:
        session['username'] = username
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'message': 'Logged out'})

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    ai_reply = f"You said: {user_message}"
    return jsonify({'reply': ai_reply})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
 {
  "username": "testuser",
  "password": "testpassword"
 }
