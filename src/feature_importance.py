import os
import joblib
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

LOGISTIC_MODEL_PATH = "models/logistic_regression_churn_model.joblib"
RF_MODEL_PATH = "models/random_forest_churn_model.joblib"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)


# ---------------------------------------------------------
# Function to get feature names
# ---------------------------------------------------------

def get_feature_names(preprocessor):
    return preprocessor.get_feature_names_out()


# =========================================================
# LOGISTIC REGRESSION FEATURE IMPORTANCE
# =========================================================

# IMPORTANT:
# Logistic model uses the step name "preprocessing"

logistic_preprocessor = logistic_model.named_steps["preprocessing"]
logistic_classifier = logistic_model.named_steps["model"]

logistic_feature_names = get_feature_names(logistic_preprocessor)

logistic_coefficients = logistic_classifier.coef_[0]

logistic_importance = pd.DataFrame({
    "Feature": logistic_feature_names,
    "Coefficient": logistic_coefficients,
    "Absolute_Importance": abs(logistic_coefficients)
})

# Sort by absolute importance
logistic_importance = logistic_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

logistic_output = os.path.join(
    OUTPUT_DIR,
    "logistic_feature_importance_current.csv"
)

logistic_importance.to_csv(
    logistic_output,
    index=False
)


# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

# Random Forest uses the step name "preprocessor"

rf_preprocessor = rf_model.named_steps["preprocessor"]
rf_classifier = rf_model.named_steps["model"]

rf_feature_names = get_feature_names(rf_preprocessor)

rf_importances = rf_classifier.feature_importances_

rf_importance = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importances
})

# Sort from most important to least important
rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

rf_output = os.path.join(
    OUTPUT_DIR,
    "rf_feature_importance_current.csv"
)

rf_importance.to_csv(
    rf_output,
    index=False
)


# =========================================================
# DISPLAY TOP FEATURES
# =========================================================

print("\n==============================================")
print("LOGISTIC REGRESSION - TOP FEATURES")
print("==============================================")

print(
    logistic_importance[
        ["Feature", "Coefficient", "Absolute_Importance"]
    ].head(15).to_string(index=False)
)


print("\n==============================================")
print("RANDOM FOREST - TOP FEATURES")
print("==============================================")

print(
    rf_importance[
        ["Feature", "Importance"]
    ].head(15).to_string(index=False)
)


print("\n==============================================")
print("FILES CREATED")
print("==============================================")

print(logistic_output)
print(rf_output)

print("\nFeature importance analysis completed successfully.")