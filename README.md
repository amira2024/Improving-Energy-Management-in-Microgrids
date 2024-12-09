# Microgrid Energy Management Simulation
This project is a recreation of a previous group research effort in microgrid energy management, combining AI models for forecasting and decision-making with a simulated communication system between microgrids. Originally conducted as part of "Using Embedded Systems and Artificial Intelligence for Power Control and Forecasting in Multiple Microgrid Networks", this version is designed to preserve and share the methodology while improving clarity, structure, and accessibility. While not identical to the original, some parts have been altered and extended for personal curiosity and further exploration, building upon what my team initially produced.
## Overview
This project simulates microgrid energy management using artificial intelligence and embedded system concepts. It integrates forecasting models, classifiers for power management actions, and a communication system between microgrids to optimize energy usage. The system also calculates energy loss and efficiency metrics, providing visual insights into its performance.

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

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Run Forecasting Models**:
   - Linear and Polynomial Regression for solar irradiance predictions.
   - View graphs of actual vs. predicted values.
2. **Train Power Management Classifiers**:
   - Train Decision Tree and Random Forest classifiers using synthetic data.
   - Predict power management actions.
3. **Simulate Microgrid Communication**:
   - Use the integrated `microgrid_simulation.py` to simulate server-client communication.
4. **Analyze Metrics**:
   - Visualize energy loss, efficiency trends, and action distributions.


### Sample Outputs
- **Predicted Actions**:
  - `Charge Battery`
  - `Discharge Battery`
  - `Curtail Load`
  - `Export to Grid`
  - `Shift Load`

## Additional Notes
It is important that you refer to the original final report that includes credit to my previous team. Their names are listed in the report.

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.

## License
This project is open-source and available under the [MIT License](LICENSE).
