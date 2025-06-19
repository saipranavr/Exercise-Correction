import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# --- GEMINI API SETUP ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual Google AI API key.
# You can get a key from https://aistudio.google.com/app/apikey
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB_1myBPRJlEmj5XebTdsXq3hEogJL2rjA")

if API_KEY == "YOUR_API_KEY":
    print("WARNING: Using a placeholder API key. Please replace 'YOUR_API_KEY' in app.py with your actual Google AI API key.")
else:
    genai.configure(api_key=API_KEY)

model = None
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error initializing GenerativeModel: {e}")
    model = None

def analyze_with_gemma(landmarks):
    """
    Analyzes pose landmarks using the Gemini model to generate feedback.
    """
    if not model:
        return "Generative model not initialized. Check API key."

    # Create a detailed prompt for the model
    prompt = f"""
    You are an expert AI personal trainer. Analyze the following bicep curl data, which consists of 3D pose landmarks from MediaPipe.
    The user is performing a bicep curl with their left arm.
    Provide concise, actionable feedback based on their form.

    Landmark Data (x, y, z, visibility):
    - Left Shoulder (ID 11): {landmarks[11]}
    - Left Elbow (ID 13): {landmarks[13]}
    - Left Wrist (ID 15): {landmarks[15]}
    - Right Shoulder (ID 12): {landmarks[12]}

    Based on the positions, critique the user's form. For example, is their elbow moving too much? Is their wrist straight? Are they using their shoulder to lift?
    Keep the feedback to one short sentence.
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Could not get feedback from the AI model."


@app.route('/analyze', methods=['POST'])
def analyze_exercise():
    """
    Receives pose landmark data, sends it for analysis,
    and returns the feedback.
    """
    data = request.get_json()
    landmarks = data.get('landmarks')

    if not landmarks or len(landmarks) < 16:
        return jsonify({'error': 'Insufficient landmark data provided'}), 400

    feedback = analyze_with_gemma(landmarks)

    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    # Note: For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(debug=True, port=5001)
