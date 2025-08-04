import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Real
from sklearn.preprocessing import MinMaxScaler

# Create an output directory to store results
output_dir = "Excel_Laplacian_Results"
os.makedirs(output_dir, exist_ok=True)

# Define the feature column labels for PIMs descriptors
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]

# Load and preprocess the PIMs descriptors
data = pd.read_csv("PIM_ExpFeatures.csv")
features = data[FEATURE_COLUMNS].values

# Normalize PIMs descriptors
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Load the Qst labels
labels_data = pd.read_csv("PIM_Qst_Labels.csv")
labels = labels_data["Qst_CO2_298K"].values  # Extract Qst labels

# Define hyperparameter search space for Bayesian Optimization (Laplacian Kernel)
space = [
    Real(1e-6, 1e2, "log-uniform", name="alpha"),  # Regularization parameter
    Real(1e-6, 1e2, "log-uniform", name="gamma")  # Kernel width
]

# Objective function for Bayesian Optimization
def objective(params):
    alpha, gamma = params
    testing_r2_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = KernelRidge(kernel="laplacian", alpha=alpha, gamma=gamma)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)
        testing_r2_scores.append(r2_score(y_test, test_predictions))

    return -np.mean(testing_r2_scores)  # Negative because skopt minimizes

# Run Bayesian Optimization
result = gp_minimize(
    objective,
    space,
    n_calls=500,
    n_initial_points=50,
    random_state=42,
    verbose=True  # Display progress
)

# Extract the best hyperparameters
best_alpha, best_gamma = result.x

# Save best hyperparameters to a CSV file
best_params = pd.DataFrame({"Alpha": [best_alpha], "Gamma": [best_gamma]})
best_params.to_csv(os.path.join(output_dir, "Excel_Laplacian_best_hyperparameters.csv"), index=False)

# Evaluate the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
training_r2_scores = []
testing_r2_scores = []
all_train_actuals, all_train_preds = [], []
all_test_actuals, all_test_preds = [], []

for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = KernelRidge(kernel="laplacian", alpha=best_alpha, gamma=best_gamma)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    all_train_actuals.extend(y_train)
    all_train_preds.extend(train_predictions)
    all_test_actuals.extend(y_test)
    all_test_preds.extend(test_predictions)

    training_r2_scores.append(r2_score(y_train, train_predictions))
    testing_r2_scores.append(r2_score(y_test, test_predictions))

# Save cross-validation R² scores
cv_results = pd.DataFrame({
    "Fold": range(1, len(training_r2_scores) + 1),
    "Training R^2": training_r2_scores,
    "Testing R^2": testing_r2_scores
})
cv_results.to_csv(os.path.join(output_dir, "Excel_Laplacian_cv_results.csv"), index=False)

# Compute average R²
average_training_r2 = np.mean(training_r2_scores)
average_testing_r2 = np.mean(testing_r2_scores)



