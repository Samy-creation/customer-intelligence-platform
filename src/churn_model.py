import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

df = pd.read_csv("data/processed/customer_features.csv")

def prepare_data(df):
    le = LabelEncoder()
    df["Country_encoded"] = le.fit_transform(df["Country"])
    
    features = ["Frequency", "Monetary", "AvgBasket", "ReturnCount", "Country_encoded", "TenureDays"]
    X = df[features]
    y = df["Churned"]
    
    return X, y

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model
    



if __name__ == "__main__":
    df = pd.read_csv("data/processed/customer_features.csv")
    X, y = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
