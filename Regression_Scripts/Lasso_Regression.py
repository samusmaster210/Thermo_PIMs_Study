import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real

# Output directory
output_dir = "Excel_Lasso_Bayes_Results"
os.makedirs(output_dir, exist_ok=True)

# Load and normalize features
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]
data = pd.read_csv("PIM_ExpFeatures.csv")
features = data[FEATURE_COLUMNS].values
labels = pd.read_csv("PIM_Qst_Labels.csv")["Qst_CO2_298K"].values

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Define optimization space
space = [Real(1e-6, 1e2, prior="log-uniform", name="alpha")]

def objective(params):
    alpha = params[0]
    r2_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))
    return -np.mean(r2_scores)

# Run Bayesian optimization
result = gp_minimize(objective, space, n_calls=500, n_initial_points=50, random_state=42, verbose=True)
best_alpha = result.x[0]

# Save best alpha
pd.DataFrame({"Best Alpha": [best_alpha]}).to_csv(os.path.join(output_dir, "Lasso_best_alpha.csv"), index=False)

# Final evaluation using best alpha
model = Lasso(alpha=best_alpha, max_iter=10000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_r2s, test_r2s = [], []

for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model.fit(X_train, y_train)
    train_r2s.append(r2_score(y_train, model.predict(X_train)))
    test_r2s.append(r2_score(y_test, model.predict(X_test)))

cv_df = pd.DataFrame({
    "Fold": range(1, 6),
    "Training R^2": train_r2s,
    "Testing R^2": test_r2s
})
cv_df.to_csv(os.path.join(output_dir, "Lasso_cv_results.csv"), index=False)
