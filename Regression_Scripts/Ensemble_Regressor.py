import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge

from skopt import gp_minimize
from skopt.space import Real

import os
os.environ['PYTHONHASHSEED'] = '42'

# Output setup
output_dir = "TEST_Excel_Ensemble_Weight_Opt_Results"
os.makedirs(output_dir, exist_ok=True)

# Load data
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]
features = pd.read_csv("PIM_ExpFeatures.csv")[FEATURE_COLUMNS].values
labels = pd.read_csv("PIM_Qst_Labels.csv")["Qst_CO2_298K"].values

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Best hyperparameters from previous Bayes optimization
xgb_model = XGBRegressor(
    n_estimators=109,
    max_depth=18,
    learning_rate=0.179388327,
    subsample=0.5,
    colsample_bytree=1,
    objective="reg:squarederror",
    n_jobs=1,
    tree_method="exact",
    verbosity=0,
    random_state=42,
    enable_categorical=False
)

lasso_model = Lasso(alpha=0.000678645)

krr_model = KernelRidge(kernel="rbf", alpha=1.0e-6, gamma=0.000483072)

# Reparameterized search space: w1 in [0,1], w2 in [0,1 - w1]
space = [Real(0.0, 1.0, name="w1"), Real(0.0, 1.0, name="w2")]

# Objective function
def objective(params):
    w1, w2 = params
    if w1 + w2 > 1.0:
        return 1e6  # Penalize infeasible region

    w3 = 1.0 - w1 - w2
    r2_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        xgb_model.fit(X_train, y_train)
        lasso_model.fit(X_train, y_train)
        krr_model.fit(X_train, y_train)

        preds = (
            w1 * xgb_model.predict(X_test) +
            w2 * lasso_model.predict(X_test) +
            w3 * krr_model.predict(X_test)
        )

        r2_scores.append(r2_score(y_test, preds))

    return -np.mean(r2_scores)  

# Run Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=1500,
    n_initial_points=250,
    random_state=42,
    verbose=True
)

# Extract best weights
best_w1, best_w2 = result.x
best_w3 = 1.0 - best_w1 - best_w2
best_weights = {"XGBoost": best_w1, "Lasso": best_w2, "RBF_KRR": best_w3}
pd.DataFrame([best_weights]).to_csv(os.path.join(output_dir, "ensemble_best_weights.csv"), index=False)

# Evaluate ensemble with best weights
train_r2s, test_r2s = [], []
all_train_actuals, all_train_preds = [], []
all_test_actuals, all_test_preds = [], []

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    xgb_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    krr_model.fit(X_train, y_train)

    train_preds = (
        best_w1 * xgb_model.predict(X_train) +
        best_w2 * lasso_model.predict(X_train) +
        best_w3 * krr_model.predict(X_train)
    )
    test_preds = (
        best_w1 * xgb_model.predict(X_test) +
        best_w2 * lasso_model.predict(X_test) +
        best_w3 * krr_model.predict(X_test)
    )

    all_train_actuals.extend(y_train)
    all_train_preds.extend(train_preds)
    all_test_actuals.extend(y_test)
    all_test_preds.extend(test_preds)

    train_r2s.append(r2_score(y_train, train_preds))
    test_r2s.append(r2_score(y_test, test_preds))

# Save results
cv_df = pd.DataFrame({
    "Fold": range(1, 6),
    "Training R^2": train_r2s,
    "Testing R^2": test_r2s
})
cv_df.to_csv(os.path.join(output_dir, "ensemble_cv_results.csv"), index=False)