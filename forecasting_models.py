import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Load the synthetic data
data_path = 'synthetic_weather_solar_data.csv'  # Replace with the actual path if different
data = pd.read_csv(data_path)

# Feature: Hour of the day (scaled as time)
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

# Visualization for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(X_test, y_pred_test_linear, color='red', label='Predicted (Linear Regression)', alpha=0.8)
plt.title('Linear Regression: Solar Irradiance Prediction')
plt.xlabel('Time (Hour)')
plt.ylabel('Solar Irradiance (W/m²)')
plt.legend()
plt.grid(True)
plt.show()

# Polynomial Regression for different degrees
degrees = [2, 3, 4, 5]
for deg in degrees:
    # Create and fit polynomial regression model
    poly_model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred_test_poly = poly_model.predict(X_test)

    # Visualization for Polynomial Regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.6)
    plt.scatter(X_test, y_pred_test_poly, color='green', label=f'Predicted (Polynomial Degree {deg})', alpha=0.8)
    plt.title(f'Polynomial Regression (Degree {deg}): Solar Irradiance Prediction')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Solar Irradiance (W/m²)')
    plt.legend()
    plt.grid(True)
    plt.show()
