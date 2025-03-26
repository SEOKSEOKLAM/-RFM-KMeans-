import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

def cluster(input_path: str, output_path: str):
    rfm = pd.read_csv(input_path, index_col=0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
    best_k, best_score = 2, -1
    for k in range(2,7):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"K={k}, Silhouette={score:.4f}")
        if score>best_score:
            best_k, best_score = k, score
    rfm['Cluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(X)
    rfm.to_csv(output_path)
    plt.figure(); plt.hist(rfm['Cluster']); plt.title("Cluster Distribution"); plt.savefig("images/cluster_distribution.png")
    plt.figure(); rfm.drop(columns=['Cluster']).boxplot(); plt.title("RFM Distribution"); plt.savefig("images/rfm_distribution.png")
    print(f"Clusters saved to {output_path}")

if __name__=="__main__":
    cluster("data/customer_rfm.csv", "data/customer_rfm_clusters.csv")
