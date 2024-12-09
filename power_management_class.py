from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 1: Generate Synthetic Data for Classifier
# Features: Solar generation, load, temperature, battery SOC
np.random.seed(42)
n_samples = 1000
solar_generation = np.random.uniform(0, 800, n_samples)  # W/m²
load_demand = np.random.uniform(0, 1000, n_samples)  # W
temperature = np.random.uniform(-10, 35, n_samples)  # °C
battery_soc = np.random.uniform(0, 100, n_samples)  # %

# Labels: Actions based on rules
# Define labels based on conditions for simplicity
labels = []
for i in range(n_samples):
    if solar_generation[i] > load_demand[i] and battery_soc[i] < 80:
        labels.append('Charge Battery')
    elif solar_generation[i] < load_demand[i] and battery_soc[i] > 20:
        labels.append('Discharge Battery')
    elif solar_generation[i] < load_demand[i] and battery_soc[i] <= 20:
        labels.append('Curtail Load')
    elif solar_generation[i] > load_demand[i] * 1.5:
        labels.append('Export to Grid')
    else:
        labels.append('Shift Load')

# Combine features into a DataFrame
features = pd.DataFrame({
    'Solar Generation (W/m²)': solar_generation,
    'Load Demand (W)': load_demand,
    'Temperature (°C)': temperature,
    'Battery SOC (%)': battery_soc
})
labels = np.array(labels)

# Step 2: Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 3: Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_dt = dt_classifier.predict(X_test)

# Classification Report
print("Decision Tree Classifier Report:")
print(classification_report(y_test, y_pred_dt))

# Visualization: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(dt_classifier, X_test, y_test, cmap="Blues", values_format='d')
plt.title("Confusion Matrix: Decision Tree Classifier")
plt.show()

# Step 4: Train Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_rf = rf_classifier.predict(X_test)

# Classification Report
print("\nRandom Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

# Visualization: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_classifier, X_test, y_test, cmap="Greens", values_format='d')
plt.title("Confusion Matrix: Random Forest Classifier")
plt.show()
