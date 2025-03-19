import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 讀取數據
data = pd.read_csv("live_streaming_sales_data.csv")

# 數據預處理
def preprocess_data(data):
    data.dropna(inplace=True)  # 移除缺失值
    data['Date'] = pd.to_datetime(data['Date'])  # 轉換日期格式
    return data

data = preprocess_data(data)

# 計算 RFM 指標
def compute_rfm(data):
    rfm = data.groupby('CustomerID').agg({
        'Date': lambda x: (data['Date'].max() - x.max()).days,  # Recency
        'CustomerID': 'count',  # Frequency
        'Price': 'sum'  # Monetary
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

rfm = compute_rfm(data)

# 數據標準化
def scale_features(rfm):
    scaler = StandardScaler()
    return scaler.fit_transform(rfm)

rfm_scaled = scale_features(rfm)

# KMeans 聚類
def apply_kmeans(rfm_scaled, rfm, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm

rfm = apply_kmeans(rfm_scaled, rfm)

# 聚類效果評估
def evaluate_clustering(rfm_scaled, rfm):
    sil_score = silhouette_score(rfm_scaled, rfm['Cluster'])
    print(f'Silhouette Score: {sil_score:.4f}')

evaluate_clustering(rfm_scaled, rfm)

# RFM 數據可視化
def plot_rfm_distribution(rfm):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rfm.drop(columns=['Cluster']))
    plt.title("RFM Value Distribution")
    plt.show()

plot_rfm_distribution(rfm)

# 聚類結果可視化
def plot_cluster_distribution(rfm):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=rfm['Cluster'], palette='viridis')
    plt.title("Customer Clusters Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()

plot_cluster_distribution(rfm)

# 代金券推薦策略
def recommend_coupon(cluster):
    if cluster == 0:
        return "100元 高價值客戶優惠券"
    elif cluster == 1:
        return "50元 中等價值客戶優惠券"
    elif cluster == 2:
        return "20元 低價值客戶優惠券"
    else:
        return "30元 首次購買優惠券"

rfm['Coupon_Recommendation'] = rfm['Cluster'].apply(recommend_coupon)

# 保存結果
def save_results(rfm, filename="customer_rfm_clusters.csv"):
    rfm.to_csv(filename, index=True)

save_results(rfm)

# 顯示部分數據
def display_sample(rfm):
    print(rfm.head())

display_sample(rfm)
