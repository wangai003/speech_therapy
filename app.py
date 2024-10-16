from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import openai
import io
import os 

app = Flask(__name__)
socketio = SocketIO(app)

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio_message')
def handle_audio_message(data):
    # Receive audio data from the client
    audio_data = data['audio']

    # Transcribe the audio using Whisper (simulated as we don't have Whisper API access)
    # In a real scenario, you would send this audio data to Whisper API
    transcript = transcribe_audio(audio_data)

    # Generate a response based on the transcript
    response = generate_response(transcript)

    # Send the response back to the client
    emit('text_response', {'text': response})

def transcribe_audio(audio_data):
    # Simulate transcription (replace this with actual Whisper API call)
    return "Simulated transcription of audio."

def generate_response(transcript):
    # Use GPT-4 to generate a response based on the transcript
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Act as a speech therapist and respond to this: {transcript}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    socketio.run(app, debug=True)
