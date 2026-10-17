import joblib

logistic_model = joblib.load("models/logistic_regression_churn_model.joblib")
rf_model = joblib.load("models/random_forest_churn_model.joblib")

print("\n--- LOGISTIC REGRESSION MODEL ---")
print(type(logistic_model))
print("Named steps:")
print(logistic_model.named_steps)

print("\n--- RANDOM FOREST MODEL ---")
print(type(rf_model))
print("Named steps:")
print(rf_model.named_steps)