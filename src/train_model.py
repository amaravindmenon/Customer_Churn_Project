import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_excel("data/client_customer_data.xlsx")

print("Dataset loaded")
print("Rows:", len(data))
print("Columns:", len(data.columns))


# --------------------------------------------------
# 2. Separate input and target
# --------------------------------------------------

X = data.drop(columns=["Churned", "Customer_ID"])

y = data["Churned"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 3. Identify column types
# --------------------------------------------------

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# --------------------------------------------------
# 5. Create ML model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


# --------------------------------------------------
# 6. Combine preprocessing + model
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# --------------------------------------------------
# 7. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 9. Evaluate
# --------------------------------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, predictions))

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print("ROC-AUC:", roc_auc)


# --------------------------------------------------
# 10. Save trained model
# --------------------------------------------------

joblib.dump(
    pipeline,
    "models/logistic_regression_churn_model.joblib"
)

print("\nModel saved successfully!")
print("models/logistic_regression_churn_model.joblib")