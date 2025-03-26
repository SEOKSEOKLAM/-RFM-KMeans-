## 1. 專案背景與目標
- **背景**：隨著直播帶貨成為熱門銷售模式，如何精準細分客戶、提高回購率與客戶忠誠度成為平台競爭的關鍵。
- **目標**：利用 RFM 模型結合 KMeans 聚類方法，對 541,910 筆直播訂單進行客戶細分，並根據不同客群特徵推薦個性化優惠券，以提升銷售及客戶回購率。

## 2. 資料說明
- **數據來源**：直播帶貨訂單數據，包含日期、價格、購買次數、客戶編號等關鍵欄位。
- **資料量**：共 541,910 筆訂單記錄。
- **主要欄位**：
  - `Date`：訂單日期。
  - `Price`：消費金額。
  - `CustomerID`：客戶唯一識別碼。
  - 其他：商品、數量、國家等。

## 3. 方法
### 3.1 RFM 模型
- **Recency (R)**：距離最近一次購買的天數。
- **Frequency (F)**：客戶購買次數。
- **Monetary (M)**：總消費金額（經過 winsorization 與 log 轉換以降低極端值影響）。

### 3.2 聚類分析（KMeans）
- **流程**：
  - 對 RFM 指標進行 MinMax 標準化。
  - 使用 KMeans 嘗試不同的聚類數（K=2 至 6）。
  - 以 Silhouette Score 評估聚類效果，選出最佳 K 值（本案例 K=2，Silhouette Score=0.4845）。
- **結果**：根據聚類結果將客戶分成兩群，並為不同群體推薦不同面額的優惠券。

### 3.3 優惠券推薦策略
- **邏輯**：
  - **高價值客戶 (Cluster 0)**：推薦 100 元代金券，其回購率及總消費額顯著較高。
  - **中低價值客戶 (Cluster 1)**：推薦 50 元代金券，使用率與回購率相對較低。
  - **首次購買客戶**：針對首次購買或潛在客戶，推薦 30 元首購優惠券，以吸引其首次消費並建立客戶黏性。
- **後續評估**：追蹤優惠券的使用情況、回購率及總銷售額，以便動態優化策略。
## 4. 系統流程與程式碼架構
- **資料清洗與預處理**：`src/preprocess.py`
- **計算 RFM 指標**：`src/compute_rfm.py`
- **聚類與可視化**：`src/clustering.py`
  - 產出圖檔：`images/cluster_distribution.png`（客戶分群分布圖）、`images/rfm_distribution.png`（RFM 箱型圖）
- **優惠券推薦**：`src/coupon_recommendation.py`
- **評估與追蹤**：`src/evaluation.py`
  - 產出圖檔：`images/performance_metrics.png`
- **資料與結果輸出**：存於 `data/` 資料夾

## 5. 結果分析
### 5.1 客戶分群 (Cluster Distribution)
- 從 `cluster_distribution.png` 可看出，主要客戶集中在 Cluster 0，代表高價值群體；而少數客戶落在 Cluster 1，消費較低。
  
### 5.2 RFM 特徵分布 (RFM Distribution)
- 箱型圖顯示各 RFM 指標的分布狀況，證實大部分客戶消費與購買頻次較低，但存在少量高價值客戶形成長尾效應。

### 5.3 優惠券策略效果 (Performance Metrics)
- 從 `performance_metrics.png` 分析，各群組優惠券使用率、回購率與總消費金額差異明顯：
  - **Cluster 0**：使用率約 30%、回購率達 100%、總消費額顯著較高。
  - **Cluster 1**：使用率略低，回購率較低，總消費額相對較低。
  
## 6. 行銷建議與未來方向
- **行銷建議**：
  - 對高價值客戶持續加大優惠券投放與個性化服務；
  - 針對中低價值客戶調整優惠券策略，提升轉化率與回購率。
- **未來方向**：
  - 引入更多客戶行為特徵（如社交互動、瀏覽歷史）；
  - 進行 A/B 測試，動態調整優惠券面額與投放時間；
  - 擴展模型至預測性行銷效果，以提升精準度。
