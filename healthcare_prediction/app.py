from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Collecting data from the form
    data = [float(request.form['Age']),
            request.form['Gender'],
            request.form['Medical Condition'],
            request.form['Insurance Provider'],
            request.form['Test Results'],
            request.form['Medication'],
            int(request.form['Days in hospital'])]

    # Feature encoding, scaling, or other preprocessing if needed
    input_data = pd.DataFrame([data], columns=['Age', 'Gender', 'Medical Condition', 'Insurance Provider', 'Test Results', 'Medication', 'Days in hospital'])

    # Assuming one-hot encoding and scaling is required like in the notebook
    input_data = pd.get_dummies(input_data)
    sscaler = StandardScaler()

    input_data_scaled = sscaler.fit_transform(input_data)

    # Predict the billing amount
    prediction = model.predict(input_data_scaled)
    
    return render_template('index.html', prediction_text=f'Predicted Billing Amount: ${prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)