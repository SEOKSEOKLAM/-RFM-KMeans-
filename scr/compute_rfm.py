import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def compute_rfm(input_path: str, output_path: str):
    df = pd.read_csv(input_path, parse_dates=['Date'])
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (df['Date'].max() - x.max()).days,
        'CustomerID': 'count',
        'Price': 'sum'
    }).rename(columns={'CustomerID':'Frequency', 'Price':'Monetary', 'Date':'Recency'})
    rfm['Monetary'] = winsorize(rfm['Monetary'], limits=[0.01, 0.01])
    rfm['Monetary'] = np.log1p(rfm['Monetary'])
    rfm.to_csv(output_path)
    print(f"RFM saved to {output_path}")

if __name__ == "__main__":
    compute_rfm("data/cleaned_data.csv", "data/customer_rfm.csv")
