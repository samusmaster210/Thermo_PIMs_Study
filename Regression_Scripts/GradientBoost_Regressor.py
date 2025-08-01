import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
import shap

# Output directory
output_dir = "Excel_GradientBoosting_Results"
os.makedirs(output_dir, exist_ok=True)

# Feature columns from the Excel descriptor file
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]

# Load and normalize features
data = pd.read_csv("PIM_ExpFeatures.csv")
features = data[FEATURE_COLUMNS].values
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Load labels
labels = pd.read_csv("PIM_Qst_Labels.csv")["Qst_CO2_298K"].values

# Define hyperparameter space
space = [
    Integer(50, 500, name="n_estimators"),
    Integer(2, 10, name="max_depth"),
    Real(0.01, 0.5, name="learning_rate"),
    Real(0.01, 1.0, name="subsample")
]

# Objective function
def objective(params):
    n_estimators, max_depth, learning_rate, subsample = params
    r2_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))

    return -np.mean(r2_scores)  # skopt minimizes

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
best_n_estimators, best_max_depth, best_learning_rate, best_subsample = result.x
best_params = pd.DataFrame({
    "n_estimators": [best_n_estimators],
    "max_depth": [best_max_depth],
    "learning_rate": [best_learning_rate],
    "subsample": [best_subsample]
})
best_params.to_csv(os.path.join(output_dir, "GBR_best_hyperparameters.csv"), index=False)

# Evaluate final model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_r2s, test_r2s = [], []
all_train_actuals, all_train_preds = [], []
all_test_actuals, all_test_preds = [], []

for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = GradientBoostingRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        subsample=best_subsample,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_r2s.append(r2_score(y_train, train_preds))
    test_r2s.append(r2_score(y_test, test_preds))

    all_train_actuals.extend(y_train)
    all_train_preds.extend(train_preds)
    all_test_actuals.extend(y_test)
    all_test_preds.extend(test_preds)

# Save cross-validation scores
cv_results = pd.DataFrame({
    "Fold": range(1, 6),
    "Training R^2": train_r2s,
    "Testing R^2": test_r2s
})
cv_results.to_csv(os.path.join(output_dir, "GBR_cv_results.csv"), index=False)

# Generate parity plots
for dataset, actuals, preds, avg_r2, filename in zip(
    ["Training", "Testing"],
    [all_train_actuals, all_test_actuals],
    [all_train_preds, all_test_preds],
    [np.mean(train_r2s), np.mean(test_r2s)],
    ["GBR_training_parity_plot.png", "GBR_testing_parity_plot.png"]
):
    plt.figure()
    plt.scatter(actuals, preds, alpha=0.7)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{dataset} Parity Plot ($R^2 = {avg_r2:.3f}$)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"Results saved in {output_dir}")

# Retrain best CatBoost model on full dataset
final_model = GradientBoostingRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        subsample=best_subsample,
        random_state=42
    )
final_model.fit(features, labels)

# SHAP analysis
explainer = shap.Explainer(final_model)
shap_values = explainer(features)

# Save SHAP values (one row per sample, one column per feature)
shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_COLUMNS)
shap_df["Sample_Index"] = np.arange(len(shap_df))
shap_df.to_csv(os.path.join(output_dir, "GradientBoosting_SHAP_values.csv"), index=False)

# Compute and save feature ranking (mean absolute SHAP)
mean_abs_shap = np.abs(shap_df[FEATURE_COLUMNS]).mean().sort_values(ascending=False)
mean_abs_shap_df = mean_abs_shap.reset_index()
mean_abs_shap_df.columns = ["Feature", "Mean_Absolute_SHAP"]
mean_abs_shap_df.to_csv(os.path.join(output_dir, "GradientBoosting_SHAP_feature_ranking.csv"), index=False)

