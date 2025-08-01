import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.preprocessing import MinMaxScaler
import shap

# Output directory
output_dir = "Excel_CatBoost_Results"
os.makedirs(output_dir, exist_ok=True)

# Load feature data
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]
data = pd.read_csv("PIM_ExpFeatures.csv")
features = data[FEATURE_COLUMNS].values

# Normalize descriptors
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Load target values
labels = pd.read_csv("PIM_Qst_Labels.csv")["Qst_CO2_298K"].values

# Define Bayesian Optimization search space
space = [
    Integer(100, 500, name="iterations"),
    Integer(3, 10, name="depth"),
    Real(0.01, 0.3, name="learning_rate"),
    Real(1, 10, name="l2_leaf_reg")
]

# Objective function
def objective(params):
    iterations, depth, learning_rate, l2_leaf_reg = params
    r2_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function="RMSE",
            verbose=0,
            random_state=42
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

# Extract best parameters
best_iterations, best_depth, best_learning_rate, best_l2 = result.x
best_params = pd.DataFrame({
    "iterations": [best_iterations],
    "depth": [best_depth],
    "learning_rate": [best_learning_rate],
    "l2_leaf_reg": [best_l2]
})
print(best_params)
best_params.to_csv(os.path.join(output_dir, "CatBoost_best_hyperparameters.csv"), index=False)

# Evaluate final model with 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
training_r2, testing_r2 = [], []
train_actuals, train_preds = [], []
test_actuals, test_preds = [], []

for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = CatBoostRegressor(
        iterations=best_iterations,
        depth=best_depth,
        learning_rate=best_learning_rate,
        l2_leaf_reg=best_l2,
        loss_function="RMSE",
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train)

    train_p = model.predict(X_train)
    test_p = model.predict(X_test)

    training_r2.append(r2_score(y_train, train_p))
    testing_r2.append(r2_score(y_test, test_p))
    train_actuals.extend(y_train)
    train_preds.extend(train_p)
    test_actuals.extend(y_test)
    test_preds.extend(test_p)

# Save CV results
cv_results = pd.DataFrame({
    "Fold": range(1, 6),
    "Training R^2": training_r2,
    "Testing R^2": testing_r2
})
print(cv_results)

# Retrain best CatBoost model on full dataset
final_model = CatBoostRegressor(
    iterations=best_iterations,
    depth=best_depth,
    learning_rate=best_learning_rate,
    l2_leaf_reg=best_l2,
    loss_function="RMSE",
    verbose=0,
    random_state=42
)
final_model.fit(features, labels)

# SHAP analysis
explainer = shap.Explainer(final_model)
shap_values = explainer(features)

# Save SHAP values (one row per sample, one column per feature)
shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_COLUMNS)
shap_df["Sample_Index"] = np.arange(len(shap_df))
shap_df.to_csv(os.path.join(output_dir, "CatBoost_SHAP_values.csv"), index=False)

# Compute and save feature ranking (mean absolute SHAP)
mean_abs_shap = np.abs(shap_df[FEATURE_COLUMNS]).mean().sort_values(ascending=False)
mean_abs_shap_df = mean_abs_shap.reset_index()
mean_abs_shap_df.columns = ["Feature", "Mean_Absolute_SHAP"]
mean_abs_shap_df.to_csv(os.path.join(output_dir, "CatBoost_SHAP_feature_ranking.csv"), index=False)






#cv_results.to_csv(os.path.join(output_dir, "CatBoost_cv_results.csv"), index=False)