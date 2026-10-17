import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "data/client_customer_data.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Prepare target
# --------------------------------------------------

y = df["Churned"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 3. Prepare features
# --------------------------------------------------

X = df.drop(columns=["Churned", "Customer_ID"])


# --------------------------------------------------
# 4. Identify column types
# --------------------------------------------------

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

categorical_columns = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
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
# 6. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# --------------------------------------------------
# 7. Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# --------------------------------------------------
# 8. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 9. Train model
# --------------------------------------------------

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 10. Evaluate model
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_probability)

print("ROC-AUC:", roc_auc)


# --------------------------------------------------
# 11. Save model
# --------------------------------------------------

model_path = "models/random_forest_churn_model.joblib"

joblib.dump(pipeline, model_path)

print("\nRandom Forest model saved successfully!")
print(model_path)