import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import socket
import threading
import matplotlib.pyplot as plt
import time

# Step 1: Generate or Load Data
data_path = 'synthetic_weather_solar_data.csv'  # Replace with your file path
data = pd.read_csv(data_path)

# Feature: Hour of the day
data['Time (Hour)'] = data['Hour'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour + 
                                          datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute / 60)

# Target: Solar Irradiance
X = data[['Time (Hour)']]
y = data['Solar Irradiance (W/m²)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_test_linear = linear_model.predict(X_test)

# Polynomial Regression
degree = 2  # Use Degree 2 for simplicity
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_test_poly = poly_model.predict(X_test)

# Visualization for Polynomial Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test, y_pred_test_poly, color='green', label=f'Predicted (Polynomial Degree {degree})', alpha=0.8)
plt.title(f'Polynomial Regression (Degree {degree}): Solar Irradiance Prediction')
plt.xlabel('Time (Hour)')
plt.ylabel('Solar Irradiance (W/m²)')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Microgrid Communication Simulation
def microgrid_server(host, port):
    """Simulates a microgrid server receiving data from clients."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print("Server is running...")
    while True:
        conn, addr = server.accept()
        print(f"Connected to client: {addr}")
        data = conn.recv(1024).decode('utf-8')
        print(f"Received from client: {data}")
        conn.close()

def microgrid_client(host, port, prediction):
    """Simulates a microgrid client sending its predictions to the server."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    message = f"Prediction from Microgrid: {prediction:.2f} W/m²"
    client.sendall(message.encode('utf-8'))
    print(f"Client sent: {message}")
    client.close()

# Step 3: Integrated Workflow
host = '127.0.0.1'
port = 65432

# Start server in a separate thread
server_thread = threading.Thread(target=microgrid_server, args=(host, port), daemon=True)
server_thread.start()

# Simulate microgrid clients
microgrid_predictions = [y_pred_test_poly[0], y_pred_test_poly[1], y_pred_test_poly[2]]
for prediction in microgrid_predictions:
    client_thread = threading.Thread(target=microgrid_client, args=(host, port, prediction))
    client_thread.start()
    client_thread.join()

# Allow server to run for a short time to process all messages
import time
time.sleep(2)
print("Simulation complete.")

# Step 1: Prepare Features and Classifier
# Using Random Forest Classifier from the previous step
feature_columns = ['Solar Generation (W/m²)', 'Load Demand (W)', 'Temperature (°C)', 'Battery SOC (%)']

# Generate synthetic feature data (same as classifier training setup)
np.random.seed(42)
n_samples = 1000
features = pd.DataFrame({
    'Solar Generation (W/m²)': np.random.uniform(0, 800, n_samples),
    'Load Demand (W)': np.random.uniform(0, 1000, n_samples),
    'Temperature (°C)': np.random.uniform(-10, 35, n_samples),
    'Battery SOC (%)': np.random.uniform(0, 100, n_samples)
})
# Example feature input
sample_features = features.iloc[0]

# Labels for training (from prior step)
labels = []
for i in range(n_samples):
    if features['Solar Generation (W/m²)'][i] > features['Load Demand (W)'][i] and features['Battery SOC (%)'][i] < 80:
        labels.append('Charge Battery')
    elif features['Solar Generation (W/m²)'][i] < features['Load Demand (W)'][i] and features['Battery SOC (%)'][i] > 20:
        labels.append('Discharge Battery')
    elif features['Solar Generation (W/m²)'][i] < features['Load Demand (W)'][i] and features['Battery SOC (%)'][i] <= 20:
        labels.append('Curtail Load')
    elif features['Solar Generation (W/m²)'][i] > features['Load Demand (W)'][i] * 1.5:
        labels.append('Export to Grid')
    else:
        labels.append('Shift Load')

labels = np.array(labels)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(features, labels)

# Step 2: Microgrid Communication
def microgrid_server_with_classifier(host, port, classifier, features):
    """Simulates a microgrid server receiving data and predicting actions using a classifier."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print("Server is running...")
    while True:
        conn, addr = server.accept()
        print(f"Connected to client: {addr}")
        data = conn.recv(1024).decode('utf-8')
        print(f"Received data from client: {data}")

        # Convert received data into features and predict action
        feature_values = list(map(float, data.split(',')))
        feature_df = pd.DataFrame([feature_values], columns=features.columns)
        action = classifier.predict(feature_df)[0]
        print(f"Predicted action: {action}")

        conn.close()


def microgrid_client_with_classifier(host, port, features):
    """Simulates a microgrid client sending feature data to the server."""
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    feature_data = ','.join(map(str, features.values))
    print(f"Client sending features: {feature_data}")
    client.sendall(feature_data.encode('utf-8'))
    client.close()


# Integrated Workflow
host = '127.0.0.1'
port = 65432

# Start server in a separate thread
server_thread = threading.Thread(target=microgrid_server_with_classifier, args=(host, port, rf_classifier, features), daemon=True)
server_thread.start()

# Simulate microgrid clients sending data
client_thread = threading.Thread(target=microgrid_client_with_classifier, args=(host, port, sample_features))
client_thread.start()
client_thread.join()

# Allow server to process messages
time.sleep(2)
print("Integrated simulation complete.")

import matplotlib.pyplot as plt

# Step 1: Simulate Energy Loss Metrics
def simulate_energy_loss_and_efficiency(actions, solar_gen, load_demand, battery_soc):
    """Simulates energy loss and efficiency metrics based on actions."""
    energy_loss = []
    efficiency = []
    
    for i in range(len(actions)):
        action = actions[i]
        generation = solar_gen[i]
        demand = load_demand[i]
        soc = battery_soc[i]

        if action == 'Curtail Load':
            loss = max(0, demand - generation)  # Unfulfilled load
            eff = generation / (demand + 1e-6) if demand > 0 else 1
        elif action == 'Charge Battery':
            loss = max(0, generation - demand) * 0.1  # Assume 10% loss in charging
            eff = 1 - (loss / (generation + 1e-6)) if generation > 0 else 0
        elif action == 'Discharge Battery':
            loss = max(0, demand - generation) * 0.05  # Assume 5% loss in discharging
            eff = 1 - (loss / (demand + 1e-6)) if demand > 0 else 0
        elif action == 'Export to Grid':
            loss = max(0, generation - demand) * 0.15  # Assume 15% loss in export
            eff = 1 - (loss / (generation + 1e-6)) if generation > 0 else 0
        else:  # 'Shift Load'
            loss = max(0, demand - generation) * 0.2  # Assume 20% loss in shifting
            eff = 1 - (loss / (demand + 1e-6)) if demand > 0 else 0

        energy_loss.append(loss)
        efficiency.append(eff * 100)  # Convert to percentage

    return energy_loss, efficiency

# Predict actions for all features in dataset
predicted_actions = rf_classifier.predict(features)

# Simulate energy loss and efficiency
energy_losses, efficiencies = simulate_energy_loss_and_efficiency(
    predicted_actions, 
    features['Solar Generation (W/m²)'], 
    features['Load Demand (W)'], 
    features['Battery SOC (%)']
)

# Step 2: Visualization of Metrics
# Plot Energy Loss Distribution
plt.figure(figsize=(10, 6))
plt.hist(energy_losses, bins=20, color='blue', alpha=0.7, label='Energy Loss (kW)')
plt.title('Distribution of Energy Losses')
plt.xlabel('Energy Loss (kW)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Plot Efficiency Over Samples
plt.figure(figsize=(10, 6))
plt.plot(efficiencies, color='green', label='Efficiency (%)')
plt.title('Efficiency Across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Efficiency (%)')
plt.legend()
plt.grid(True)
plt.show()

# Action Distribution (Bar Chart)
action_counts = pd.Series(predicted_actions).value_counts()
action_counts.plot(kind='bar', figsize=(10, 6), color='orange', alpha=0.8)
plt.title('Distribution of Power Management Actions')
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

