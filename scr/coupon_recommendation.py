import pandas as pd

def recommend(cluster):
    return {0:"100元 高價值",1:"50元 中價值",2:"20元 低價值"}.get(cluster, "30元 首購")

def apply(input_path: str, output_path: str):
    df = pd.read_csv(input_path, index_col=0)
    df['Coupon'] = df['Cluster'].apply(recommend)
    df.to_csv(output_path)
    print(f"Coupon recommendations saved to {output_path}")

if __name__=="__main__":
    apply("data/customer_rfm_clusters.csv", "data/customer_rfm_recommendations.csv")
