
# Microgrid Energy Management Simulation

## Overview
This project simulates microgrid energy management using artificial intelligence and embedded system concepts. It integrates forecasting models, classifiers for power management actions, and a communication system between microgrids to optimize energy usage. The system also calculates energy loss and efficiency metrics, providing visual insights into its performance..

## Features
1. **AI Forecasting Models**:
   - Linear Regression and Polynomial Regression for predicting solar irradiance and load.
2. **Power Management Classifiers**:
   - Decision Tree and Random Forest models for selecting optimal energy actions.
3. **Microgrid Communication Simulation**:
   - Simulates data exchange between microgrids using TCP/IP.
4. **Energy Loss and Efficiency Metrics**:
   - Analyzes energy loss and efficiency during power management actions.
5. **Visualization**:
   - Graphs for action distribution, energy loss, and efficiency.
## Requirements
- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `socket`
## Usage

1. Run Forecasting Models:

  - Linear and Polynomial Regression for solar irradiance predictions.

  - View graphs of actual vs. predicted values.

2. Train Power Management Classifiers:

  - Train Decision Tree and Random Forest classifiers using synthetic data.

  - Predict power management actions.

3. Simulate Microgrid Communication:

  - Use the integrated microgrid_simulation.py to simulate server-client communication.

4. Analyze Metrics:

  - Visualize energy loss, efficiency trends, and action distributions.
  
Install dependencies using:
```bash
pip install -r requirements.txt
##
