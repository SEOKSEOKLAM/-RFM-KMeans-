import pandas as pd

def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess("data/live_streaming_sales_data.csv", "data/cleaned_data.csv")
