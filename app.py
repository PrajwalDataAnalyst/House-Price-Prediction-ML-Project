from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the trained model and columns information
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    columns = json.load(f)
    data_columns = columns['data_columns']

# Define a route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    loc_index = data_columns.index(location.lower())
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    return render_template('index.html', prediction_text=f'Predicted Price: {predicted_price:.2f} lakhs')

if __name__ == "__main__":
    app.run(debug=True)
