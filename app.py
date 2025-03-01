from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "/app/uploads"
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # DEBUG: Print file path before saving
        print(f"Saving file to: {filepath}")

        file.save(filepath)

        # Check if file exists after saving
        if os.path.exists(filepath):
            print("File successfully saved!")
        else:
            print("File saving failed!")

        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    
    print("Invalid file format")
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/get_user_dates', methods=['GET'])
def get_user_dates():
    all_users = {}
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            df = pd.read_csv(filepath)
            
            if 'user' in df.columns and 'day' in df.columns:
                for _, row in df.iterrows():
                    user = row['user']
                    date = row['day']
                    
                    if user not in all_users:
                        all_users[user] = set()
                    all_users[user].add(date)
    
    # Convert set to list for JSON serialization
    all_users = {user: list(dates) for user, dates in all_users.items()}
    return jsonify({'users': all_users})

# Load LSTM Autoencoder Model
MODEL_PATH = "/app/model/best_lstm_autoencoder2.h5"
model = tf.keras.models.load_model(MODEL_PATH)
MAX_SEQ_LENGTH = 74  # Ensure consistency with training

def get_latest_file():
    """Fetches the latest uploaded file from the uploads folder."""
    files = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if not files:
        return None
    return max(files, key=os.path.getctime)  # Get the most recently modified file

def preprocess_csv(file_path):
    """Loads and preprocesses CSV data for LSTM autoencoder."""
    df = pd.read_csv(file_path)
    df['activity_encoded'] = df['activity_encoded'].apply(eval)
    X = pad_sequences(df['activity_encoded'], maxlen=MAX_SEQ_LENGTH, padding='post')
    return df, X

def get_anomaly_scores(df, X):
    """Calculates anomaly scores based on reconstruction error."""
    X = X.reshape((-1, MAX_SEQ_LENGTH, 1))  # Reshape for LSTM input
    reconstructions = model.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=(1, 2))  # Compute MSE
    
    # Min-Max Scaling to range 1-100
    min_score = np.min(mse)
    max_score = np.max(mse)
    
    if max_score == min_score:  # Prevent division by zero
        scaled_scores = np.ones_like(mse) * 50  # Default to mid-range
    else:
        scaled_scores = ((mse - min_score) / (max_score - min_score)) * 99 + 1

    df['anomaly_score'] = scaled_scores
    return df[['user', 'day', 'anomaly_score']].to_dict(orient='records')

@app.route('/predict', methods=['POST'])
def predict():
    """Loads the latest uploaded CSV, processes it, and returns anomaly scores."""
    latest_file = get_latest_file()
    if not latest_file:
        return jsonify({"error": "No uploaded files found"}), 400
    
    df, X = preprocess_csv(latest_file)
    scores = get_anomaly_scores(df, X)
    
    return jsonify({"anomaly_data": scores})

