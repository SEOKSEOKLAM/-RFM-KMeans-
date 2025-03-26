import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate(raw_path: str, rec_path: str):
    # 讀取原始清洗後資料（包含 Date, CustomerID, Price）
    df_raw = pd.read_csv(raw_path, parse_dates=['Date'])
    # 讀取聚類 + 推薦結果
    df_rec = pd.read_csv(rec_path, index_col=0)
    # 合併
    df = df_raw.merge(df_rec[['Cluster','Coupon']], left_on='CustomerID', right_index=True)

    # 隨機模擬 30% 客戶使用優惠券
    df['CouponUsed'] = np.random.choice([0,1], size=len(df), p=[0.7,0.3])

    # 計算指標
    summary = df.groupby('Cluster').agg(
        coupon_usage_rate=('CouponUsed','mean'),
        repurchase_rate=('CustomerID', lambda x: x.duplicated().mean()),
        total_spend=('Price','sum')
    )
    summary.to_csv("data/evaluation_summary.csv")

    # 確保 images 目錄存在
    os.makedirs("images", exist_ok=True)

    # 畫圖並存檔
    summary.plot.bar(subplots=True, layout=(1,3), figsize=(15,4))
    plt.tight_layout()
    plt.savefig("images/performance_metrics.png")
    plt.close()

    print("✅ Evaluation complete")
    print(" • Summary saved → data/evaluation_summary.csv")
    print(" • Chart saved   → images/performance_metrics.png")

if __name__ == "__main__":
    evaluate("data/cleaned_data.csv", "data/customer_rfm_recommendations.csv")
