import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.preprocessing import MinMaxScaler
import shap

import os
os.environ['PYTHONHASHSEED'] = '42'

# Output directory
output_dir = "Excel_XGBoost_Results"
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

# Define search space
space = [
    Integer(50, 300, name="n_estimators"),
    Integer(2, 20, name="max_depth"),
    Real(0.01, 0.3, name="learning_rate"),
    Real(0.5, 1.0, name="subsample"),
    Real(0.5, 1.0, name="colsample_bytree")
]

# Objective function
def objective(params):
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree = params
    r2_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            n_jobs=1,  # <- Enforce single-threaded determinism
            verbosity=0,
            random_state=42,
            tree_method="exact",  # <- Optional: exact split for full determinism
            enable_categorical=False  # <- In case categorical creep in
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))

    return -np.mean(r2_scores)

# Run Bayesian optimization
result = gp_minimize(
    objective,
    space,
    n_calls=500,
    n_initial_points=50,
    random_state=42,
    verbose=True
)

# Best hyperparameters
best_params = result.x
param_names = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
best_params_dict = dict(zip(param_names, best_params))
pd.DataFrame([best_params_dict]).to_csv(os.path.join(output_dir, "XGBoost_best_hyperparameters.csv"), index=False)

# Final model evaluation
train_r2s, test_r2s = [], []
all_train_actuals, all_train_preds = [], []
all_test_actuals, all_test_preds = [], []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = XGBRegressor(**best_params_dict, n_jobs=1, verbosity=0, tree_method="exact", random_state=42, enable_categorical=False)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    all_train_actuals.extend(y_train)
    all_train_preds.extend(train_preds)
    all_test_actuals.extend(y_test)
    all_test_preds.extend(test_preds)

    train_r2s.append(r2_score(y_train, train_preds))
    test_r2s.append(r2_score(y_test, test_preds))

# Save CV scores
cv_df = pd.DataFrame({
    "Fold": range(1, 6),
    "Training R^2": train_r2s,
    "Testing R^2": test_r2s
})
cv_df.to_csv(os.path.join(output_dir, "XGBoost_cv_results.csv"), index=False)