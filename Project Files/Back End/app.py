from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
try:
    with open('traffic_volume.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Please ensure 'traffic_volume.pkl' exists.")
    model = None

# Load scaler if available
try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Scaler loaded successfully!")
except FileNotFoundError:
    print("Scaler file not found. Using default StandardScaler.")
    scaler = StandardScaler()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return render_template('error.html', message="Model not loaded properly")
        
        # Get input values from form
        holiday = int(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        
        # Encode weather condition (simple encoding)
        weather_mapping = {
            'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3, 
            'Mist': 4, 'Fog': 5, 'Drizzle': 6, 'Thunderstorm': 7
        }
        weather_encoded = weather_mapping.get(weather, 1)  # Default to 'Clouds'
        
        # Create feature array
        features = np.array([[holiday, temp, rain, snow, weather_encoded, 
                            year, month, day, hour]])
        
        # Scale features if scaler is available
        try:
            features_scaled = scaler.transform(features)
        except:
            features_scaled = features
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Round prediction to reasonable number
        prediction = round(prediction, 2)
        
        # Render result template with prediction
        return render_template('result.html', 
                             prediction=prediction,
                             input_data={
                                 'holiday': 'Yes' if holiday else 'No',
                                 'temp': temp,
                                 'rain': rain,
                                 'snow': snow,
                                 'weather': weather,
                                 'year': year,
                                 'month': month,
                                 'day': day,
                                 'hour': hour
                             })
        
    except Exception as e:
        return render_template('error.html', message=f"Error occurred: {str(e)}")

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)