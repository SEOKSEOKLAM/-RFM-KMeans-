利用 RFM+KMeans 對 541,910 筆直播訂單做客戶細分，並依照分群結果推薦個性化優惠券。  
最佳聚類數 K=2（Silhouette Score=0.4845），高價值群體使用率30%、回購率100%、總消費額遠高於中價值群。  
結果存於 `data/`，可視化圖檔位於 `images/`，完整流程：`src/` → `python src/evaluation.py`。  
