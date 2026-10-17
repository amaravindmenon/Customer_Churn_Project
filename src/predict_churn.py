import pandas as pd
import joblib


# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

model = joblib.load(
    "models/churn_model.joblib"
)

print("Trained model loaded successfully!")


# --------------------------------------------------
# 2. Load customer data
# --------------------------------------------------

data = pd.read_excel(
    "data/client_customer_data.xlsx"
)


# --------------------------------------------------
# 3. Keep customer IDs
# --------------------------------------------------

customer_ids = data["Customer_ID"]


# --------------------------------------------------
# 4. Remove columns that aren't model inputs
# --------------------------------------------------

X = data.drop(
    columns=["Customer_ID", "Churned"],
    errors="ignore"
)


# --------------------------------------------------
# 5. Predict churn probability
# --------------------------------------------------

probabilities = model.predict_proba(X)[:, 1]


# --------------------------------------------------
# 6. Create results
# --------------------------------------------------

results = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Churn_Probability": probabilities
})


# --------------------------------------------------
# 7. Convert probability into risk
# --------------------------------------------------

def risk_level(probability):

    if probability < 0.30:
        return "Low"

    elif probability < 0.60:
        return "Medium"

    else:
        return "High"


results["Risk_Level"] = results[
    "Churn_Probability"
].apply(risk_level)


# --------------------------------------------------
# 8. Sort highest risk first
# --------------------------------------------------

results = results.sort_values(
    "Churn_Probability",
    ascending=False
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nTop customers at risk:\n")

print(
    results.head(20).to_string(index=False)
)


# --------------------------------------------------
# 10. Save predictions
# --------------------------------------------------

results.to_csv(
    "outputs/customer_churn_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to:"
    " outputs/customer_churn_predictions.csv"
)