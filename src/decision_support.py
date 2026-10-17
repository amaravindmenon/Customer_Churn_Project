import os
import joblib
import pandas as pd


# =========================================================
# PATHS
# =========================================================

DATA_PATH = "data/client_customer_data.xlsx"
MODEL_PATH = "models/logistic_regression_churn_model.joblib"
EXPLANATION_PATH = "outputs/local_explanations_high_risk.csv"

OUTPUT_PATH = "outputs/decision_support.csv"
HIGH_RISK_OUTPUT = "outputs/high_risk_retention_priorities.csv"

os.makedirs("outputs", exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.60


# =========================================================
# LOAD DATA, MODEL AND EXPLANATIONS
# =========================================================

df = pd.read_excel(DATA_PATH)

model = joblib.load(MODEL_PATH)

explanations = pd.read_csv(EXPLANATION_PATH)


# =========================================================
# PREPARE MODEL INPUT
# =========================================================

X = df.drop(columns=["Churned", "Customer_ID"])


# =========================================================
# GENERATE CHURN PROBABILITIES
# =========================================================

probabilities = model.predict_proba(X)[:, 1]


# =========================================================
# CREATE DECISION-SUPPORT TABLE
# =========================================================

decision = df.copy()

decision["Churn_Probability"] = probabilities


# =========================================================
# ASSIGN RISK LEVEL
# =========================================================

def assign_risk(probability):

    if probability < LOW_THRESHOLD:
        return "Low"

    elif probability < HIGH_THRESHOLD:
        return "Medium"

    else:
        return "High"


decision["Risk_Level"] = decision["Churn_Probability"].apply(
    assign_risk
)


# =========================================================
# ASSIGN RETENTION PRIORITY
# =========================================================

def assign_priority(risk):

    if risk == "High":
        return "Immediate"

    elif risk == "Medium":
        return "Planned"

    else:
        return "Monitor"


decision["Retention_Priority"] = decision["Risk_Level"].apply(
    assign_priority
)


# =========================================================
# MERGE LOCAL EXPLANATIONS
# =========================================================

decision = decision.merge(
    explanations[
        [
            "Customer_ID",
            "Top_Risk_Factors",
            "Top_Protective_Factors"
        ]
    ],
    on="Customer_ID",
    how="left"
)


# =========================================================
# RECOMMENDED ACTION
# =========================================================

def recommend_action(row):

    risk = row["Risk_Level"]

    risk_factors = str(row["Top_Risk_Factors"]).lower()

    # High-risk customers receive targeted actions
    if risk == "High":

        if "days_since_last_activity" in risk_factors:
            return "Priority re-engagement and retention contact"

        elif "support_tickets_90d" in risk_factors:
            return "Priority customer-support follow-up"

        elif "satisfaction_score" in risk_factors:
            return "Customer satisfaction follow-up"

        elif "contract_type_monthly" in risk_factors:
            return "Priority retention discussion and contract review"

        else:
            return "Priority retention contact"

    # Medium-risk customers
    elif risk == "Medium":

        if "days_since_last_activity" in risk_factors:
            return "Planned re-engagement campaign"

        elif "support_tickets_90d" in risk_factors:
            return "Planned customer-support follow-up"

        else:
            return "Monitor and include in retention campaign"

    # Low-risk customers
    else:
        return "Routine monitoring"


decision["Recommended_Action"] = decision.apply(
    recommend_action,
    axis=1
)


# =========================================================
# ROUND PROBABILITY FOR READABILITY
# =========================================================

decision["Churn_Probability"] = decision[
    "Churn_Probability"
].round(4)


# =========================================================
# SORT BY PRIORITY
# =========================================================

priority_order = {
    "Immediate": 1,
    "Planned": 2,
    "Monitor": 3
}

decision["Priority_Order"] = decision[
    "Retention_Priority"
].map(priority_order)

decision = decision.sort_values(
    by=["Priority_Order", "Churn_Probability"],
    ascending=[True, False]
)


# =========================================================
# REMOVE INTERNAL SORTING COLUMN
# =========================================================

decision = decision.drop(
    columns=["Priority_Order"]
)


# =========================================================
# SAVE COMPLETE DECISION-SUPPORT FILE
# =========================================================

decision.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# CREATE HIGH-RISK RETENTION LIST
# =========================================================

high_risk = decision[
    decision["Risk_Level"] == "High"
].copy()


# Keep the most useful management columns
high_risk = high_risk[
    [
        "Customer_ID",
        "Churn_Probability",
        "Risk_Level",
        "Retention_Priority",
        "Top_Risk_Factors",
        "Top_Protective_Factors",
        "Recommended_Action"
    ]
]


high_risk.to_csv(
    HIGH_RISK_OUTPUT,
    index=False
)


# =========================================================
# DISPLAY SUMMARY
# =========================================================

print("\n==============================================")
print("DECISION SUPPORT ANALYSIS")
print("==============================================")


print("\nRisk distribution:")

print(
    decision["Risk_Level"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
    .to_string()
)


print("\nRetention priority:")

print(
    decision["Retention_Priority"]
    .value_counts()
    .reindex(
        ["Immediate", "Planned", "Monitor"],
        fill_value=0
    )
    .to_string()
)


print("\n==============================================")
print("TOP 15 HIGH-RISK CUSTOMERS")
print("==============================================")


print(
    high_risk.head(15).to_string(index=False)
)


print("\n==============================================")
print("FILES CREATED")
print("==============================================")

print(OUTPUT_PATH)
print(HIGH_RISK_OUTPUT)

print("\nDecision-support analysis completed successfully.")