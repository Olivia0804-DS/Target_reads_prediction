from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a CSV file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    # Save the file to a temporary location
    filepath = os.path.join(os.getcwd(), file.filename)
    file.save(filepath)
    
    # Load the CSV file into a DataFrame
    input_data = pd.read_csv(filepath)
    
    # Preprocess and predict using the model
    prediction = model.predict(input_data)
    
    # Map the prediction results: 0 -> "reach 1.2", 1 -> "didn't reach 1.2"
    prediction_mapped = [1 if pred == 0 else 0 for pred in prediction]
    
    # Clean up the file after reading
    os.remove(filepath)
    
    # Render the template with the prediction result
    return render_template('index.html', prediction=prediction_mapped)

if __name__ == '__main__':
    app.run(debug=True)
