from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import BallTree
import numpy as np

app = Flask(__name__)

# Load the dataset for operator prediction
data = pd.read_csv("dataset.csv")

# Data Preprocessing
# Drop rows with missing values
data = data.dropna()

# One-hot encode the 'calldrop_category' column
data = pd.get_dummies(data, columns=['calldrop_category'])

# Feature Engineering
# Convert latitude and longitude to radians
data['latitude_rad'] = np.radians(data['latitude'])
data['longitude_rad'] = np.radians(data['longitude'])

# Train-Test Split
X_columns = ['latitude_rad', 'longitude_rad', 'calldrop_category_Poor Voice Quality', 'calldrop_category_Satisfactory', 'calldrop_category_Call Dropped']
X = pd.DataFrame(data, columns=X_columns)
y = data['operator']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model for operator prediction
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Load the dataset for data speed
# Read dataset2.csv
data_speed = pd.read_csv("dataset2.csv")

# Convert service providers
data_speed['Service Provider'] = data_speed['Service Provider'].replace({'JIO': 'Rjio', 'VODAFONE': 'VI', 'IDEA': 'Airtel'})

# Save the modified dataset
data_speed.to_csv("modified_dataset2.csv", index=False)

# Function to predict the best operator, inout_travelling, indoor/outdoor service, and data speed for a given location
def predict_network_info(latitude, longitude):
    # Convert latitude and longitude to radians
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)
    
    # Find the nearest known location
    known_locations = data[['latitude_rad', 'longitude_rad']].values
    tree = BallTree(known_locations, leaf_size=15, metric='haversine')
    dist, ind = tree.query([[latitude_rad, longitude_rad]], k=1)
    nearest_index = ind[0][0]
    
    # Get call drop rate for the nearest location
    call_drop_rate = data.iloc[nearest_index]['calldrop_category_Satisfactory']
    poor_voice_quality = data.iloc[nearest_index]['calldrop_category_Poor Voice Quality']
    call_dropped = data.iloc[nearest_index]['calldrop_category_Call Dropped']
    
    # Predict the best operator based on the nearest location and call drop rate
    predicted_operator = model.predict([[latitude_rad, longitude_rad, poor_voice_quality, call_drop_rate, call_dropped]])[0]
    
    # Predict the inout_travelling category based on the nearest location
    predicted_inout_travelling = data.iloc[nearest_index]['inout_travelling']
    
    # Get the state name of the nearest location
    state_name = data.iloc[nearest_index]['state_name']
    
    # Look up the data speed for the predicted operator and state name
    speed = data_speed[(data_speed['Service Provider'] == predicted_operator) & (data_speed['LSA'] == state_name)]['Data Speed(Mbps)'].max()
    
    # If speed is NaN, provide the average speed in that location
    if pd.isnull(speed):
        avg_speed = data_speed[data_speed['LSA'] == state_name]['Data Speed(Mbps)'].mean()
        return predicted_operator, predicted_inout_travelling, avg_speed
    else:
        return predicted_operator, predicted_inout_travelling, speed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    latitude = float(request.json['latitude'])
    longitude = float(request.json['longitude'])
    
    predicted_operator, predicted_inout_travelling, speed = predict_network_info(latitude, longitude)
    
    return jsonify({'operator': predicted_operator, 'inout_travelling': predicted_inout_travelling, 'speed': speed})

if __name__ == "__main__":
    app.run(debug=True)
