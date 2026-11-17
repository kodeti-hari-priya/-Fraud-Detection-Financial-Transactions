
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------

DATA_PATH = "data/train_transactions.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Loading transaction dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())


# ------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------

df = df.drop_duplicates()

TARGET = "is_fraud"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found. "
        f"Available columns: {list(df.columns)}"
    )

if df[TARGET].isnull().any():
    raise ValueError("Target column contains missing values.")

df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("is_fraud must contain only 0 and 1.")

df[TARGET] = df[TARGET].astype(int)

if df[TARGET].nunique() != 2:
    raise ValueError("Dataset must contain both fraud and legitimate records.")


# ------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------

def prepare_features(data):

    data = data.copy()

    # Remove transaction identifier
    if "transaction_id" in data.columns:
        data = data.drop(columns=["transaction_id"])

    # Extract useful information from timestamp
    if "timestamp" in data.columns:
        timestamp = pd.to_datetime(
            data["timestamp"], errors="coerce", utc=True
        )

        data["transaction_hour"] = timestamp.dt.hour
        data["transaction_day"] = timestamp.dt.day
        data["transaction_month"] = timestamp.dt.month
        data["transaction_weekday"] = timestamp.dt.dayofweek

        data = data.drop(columns=["timestamp"])

    return data


X = prepare_features(df.drop(columns=[TARGET]))
y = df[TARGET]

FEATURE_COLUMNS = X.columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ------------------------------------------------
# 4. TRAIN AND TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------
# 5. PREPROCESSING PIPELINE
# ------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ------------------------------------------------
# 6. FIVE MACHINE LEARNING MODELS
# ------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "SVM": SVC(
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )
}


# ------------------------------------------------
# 7. MODEL TRAINING AND EVALUATION
# ------------------------------------------------

results = []
trained_models = {}

print("\nTraining machine learning models...\n")

for name, classifier in models.items():

    print("Training:", name)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "model": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

    trained_models[name] = model

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


# ------------------------------------------------
# 8. SELECT BEST MODEL
# ------------------------------------------------

best_result = max(
    results,
    key=lambda item: item["f1_score"]
)

best_model_name = best_result["model"]
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# ------------------------------------------------
# 9. SAVE MODEL AND RESULTS
# ------------------------------------------------

joblib.dump(
    best_model,
    os.path.join(ARTIFACT_DIR, "fraud_model.joblib")
)

with open(
    os.path.join(ARTIFACT_DIR, "model_metadata.json"),
    "w"
) as file:

    json.dump({
        "best_model": best_model_name,
        "target_column": TARGET,
        "feature_columns": FEATURE_COLUMNS
    }, file, indent=4)

with open(
    os.path.join(ARTIFACT_DIR, "model_metrics.json"),
    "w"
) as file:

    json.dump(results, file, indent=4)


# ------------------------------------------------
# 10. DISPLAY FINAL RESULTS
# ------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

print("\nBest Performing Model:", best_model_name)

print("\nModel and metadata saved successfully!")
print("Location:", ARTIFACT_DIR)
