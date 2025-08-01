import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.preprocessing import MinMaxScaler
import shap

# Output directory
output_dir = "Excel_ExtraTrees_Results"
os.makedirs(output_dir, exist_ok=True)

# Load PIMs descriptors and labels
FEATURE_COLUMNS = [
    "C", "H", "O", "N", "F", "S", "System_Size", "a", "b", "c",
    "density", "PLD", "LCD", "N2_SA",
    "Probe_Accessible", "Probe_Occupiable", "Rosenbluth_Weight"
]
data = pd.read_csv("PIM_ExpFeatures.csv")
features = data[FEATURE_COLUMNS].values
labels = pd.read_csv("PIM_Qst_Labels.csv")["Qst_CO2_298K"].values

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Define hyperparameter search space
space = [
    Integer(50, 300, name="n_estimators"),
    Integer(2, 20, name="max_depth"),
    Integer(2, 20, name="min_samples_split"),
    Integer(1, 10, name="min_samples_leaf")
]

# Objective function
def objective(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params
    r2_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2_scores.append(r2_score(y_test, preds))

    return -np.mean(r2_scores)  # skopt minimizes

# Run Bayesian Optimization
result = gp_minimize(
    objective,
    space,
    n_calls=500,
    n_initial_points=50,
    random_state=42,
    verbose=True
)

# Extract best parameters
best_n_estimators, best_max_depth, best_min_split, best_min_leaf = result.x
best_params = pd.DataFrame({
    "n_estimators": [best_n_estimators],
    "max_depth": [best_max_depth],
    "min_samples_split": [best_min_split],
    "min_samples_leaf": [best_min_leaf]
})
best_params.to_csv(os.path.join(output_dir, "ExtraTrees_best_hyperparameters.csv"), index=False)

# Final evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_r2s, test_r2s = [], []
all_train_actuals, all_train_preds = [], []
all_test_actuals, all_test_preds = [], []

for train_idx, test_idx in kf.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    model = ExtraTreesRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_split,
        min_samples_leaf=best_min_leaf,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

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
cv_df.to_csv(os.path.join(output_dir, "ExtraTrees_cv_results.csv"), index=False)

# Parity plots
for label, actual, pred, r2, fname in zip(
    ["Training", "Testing"],
    [all_train_actuals, all_test_actuals],
    [all_train_preds, all_test_preds],
    [np.mean(train_r2s), np.mean(test_r2s)],
    ["ExtraTrees_training_parity_plot.png", "ExtraTrees_testing_parity_plot.png"]
):
    plt.figure()
    plt.scatter(actual, pred, alpha=0.7, label=f"{label} Data")
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', label="Perfect Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{label} Parity Plot ($R^2 = {r2:.3f}$)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


# Retrain best CatBoost model on full dataset
final_model = ExtraTreesRegressor(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        min_samples_split=best_min_split,
        min_samples_leaf=best_min_leaf,
        random_state=42,
        n_jobs=1
    )
final_model.fit(features, labels)

# SHAP analysis
explainer = shap.Explainer(final_model)
shap_values = explainer(features)

# Save SHAP values (one row per sample, one column per feature)
shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_COLUMNS)
shap_df["Sample_Index"] = np.arange(len(shap_df))
shap_df.to_csv(os.path.join(output_dir, "ExtraTrees_SHAP_values.csv"), index=False)

# Compute and save feature ranking (mean absolute SHAP)
mean_abs_shap = np.abs(shap_df[FEATURE_COLUMNS]).mean().sort_values(ascending=False)
mean_abs_shap_df = mean_abs_shap.reset_index()
mean_abs_shap_df.columns = ["Feature", "Mean_Absolute_SHAP"]
mean_abs_shap_df.to_csv(os.path.join(output_dir, "ExtraTrees_SHAP_feature_ranking.csv"), index=False)

