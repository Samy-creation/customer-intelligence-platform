import pandas as pd
import numpy as np
import os

# Load and clean the raw transaction data. Use a cached CSV copy if it already exists.
def load_and_clean(path="data/raw/online_retail_II.xlsx"):
    merged_path = "data/raw/transactions_merged.csv"
    
    if os.path.exists(merged_path):
        # Read the merged transactions file when it is already available
        df = pd.read_csv(merged_path)
    
    else:
        # Otherwise load both Excel sheets and merge them into one dataframe
        x1 = pd.ExcelFile(path)
        sheet1 = x1.parse("Year 2009-2010")
        sheet2 = x1.parse("Year 2010-2011")
        df = pd.concat([sheet1, sheet2], ignore_index=True)
        df.to_csv("data/raw/transactions_merged.csv", index=False)

    # Remove rows with missing customer IDs and invalid negative prices
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Price"] >= 0]
    df["TotalPrice"] = df["Price"] * df["Quantity"]
    

    return df


# Compute RFM features for each customer: Recency, Frequency, Monetary.
def compute_rfm(df):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()
    # Mark customers as churned when their last purchase was more than 90 days ago
    rfm["Churned"] = np.where(rfm["Recency"] > 90, 1, 0)

    return rfm



if __name__ == "__main__": 
    df = load_and_clean()
    rfm = compute_rfm(df)
    rfm.to_csv("data/processed/customer_features.csv", index=False)