### Feature Analysis：Correlation / Mutual Information / ROC AUC

這三個指標都能用來探索 **Feature（X）與 Target（y）的關係**，但回答的是不同問題：

- **Correlation（相關係數）**：X 與 y 是否存在**線性關係**？方向是正還是負？
- **Mutual Information（MI，互資訊）**：知道 X 後，可以減少多少對 y 的**不確定性**？
- **ROC AUC**：把 X 當成分數時，能否把 **y=1 排在 y=0 前面**？

### 1. 快速比較

| 指標 | 衡量重點 | 數值範圍 | 能看方向？ | 能捕捉非線性？ | 常見用途 |
|---|---|---|---|---|---|
| Pearson Correlation | 線性關聯 | -1～1 | 是（正／負） | 不擅長 | 快速探索數值特徵 |
| Mutual Information | 統計依賴、資訊量 | ≥ 0 | 否 | 是 | 發掘複雜關係、特徵篩選 |
| ROC AUC | 正負類別的排序區分能力 | 0～1 | 是（高分偏向哪類） | 對原始分數的非單調關係不擅長 | 二元分類的單一特徵評估、模型評估 |

> 注意：這裡的 Correlation 是 **Pearson**；Spearman 則衡量排序上的單調關係。MI 的數值大小也會受到估計方法與資料處理影響，不適合直接當成準確率。

### 2. Correlation（Pearson）

- `+1`：完全正向線性關係。
- `-1`：完全負向線性關係。
- `0`：沒有**線性**關係，不代表完全沒有關係。

```python
# y 為二元 0/1 時，數值特徵與 y 的 Pearson correlation 也可計算
corr = df[['age', 'income', 'converted']].corr()
print(corr['converted'])
```

**限制：** 若 X 與 y 呈 U 型關係，正負趨勢可能抵銷，Correlation 接近 0。

### 3. Mutual Information（MI）

直覺：**X 能提供多少關於 y 的資訊？**

- `MI = 0`：理論上 X 與 y 獨立，沒有資訊依賴。
- `MI > 0`：存在資訊依賴；可以包含非線性、非單調關係。
- MI 不會告訴你「X 越大，y 越可能是 1」還是相反。

```python
from sklearn.feature_selection import mutual_info_classif

# 範例：X_train 已處理成 sklearn 可接受的數值矩陣
mi = mutual_info_classif(X_train, y_train, random_state=1)
```

**注意：** 類別特徵應正確編碼，並依需要設定 `discrete_features`；MI 是從有限樣本估計的，結果會受樣本量與估計方法影響。

### 4. ROC AUC（重點）

ROC AUC 衡量的是**排序能力**，不是預測分數是否接近真實機率。

- `AUC = 1`：所有正類都排在負類前面。
- `AUC = 0.5`：排序區分能力相當於隨機。
- `AUC < 0.5`：分數方向可能與正類相反。
- **分數不必介於 0～1**：年齡、收入、任意連續模型分數都可以使用。

#### ROC → AUC 計算流程

1. 將原始分數 `y_score` **由高到低排序**（`y_true` 與各樣本保持配對）。
2. 讓 **Threshold 由高到低變動**；每次將分數 ≥ Threshold 的樣本判為正類。
3. 計算各 Threshold 的：
   - `TPR = TP / (TP + FN)`（Recall）
   - `FPR = FP / (FP + TN)`
4. 以 **FPR 為 X 軸、TPR 為 Y 軸**，連成 ROC Curve。
5. 計算曲線下方的面積，即 **AUC**（可用梯形法則）。

#### 例子：分數不在 0～1 也沒關係

| 樣本 | 真實 y | 分數 |
|---|---:|---:|
| A | 1 | 90 |
| B | 0 | 70 |
| C | 1 | 60 |
| D | 0 | 20 |

| Threshold | FPR | TPR |
|---|---:|---:|
| > 90 | 0 | 0 |
| 90 | 0 | 0.5 |
| 70 | 0.5 | 0.5 |
| 60 | 0.5 | 1.0 |
| 20 | 1.0 | 1.0 |

面積 = `0.5 × 0.5 + 0.5 × 1.0 = 0.75`。

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

y_true = [1, 0, 1, 0]
y_score = [90, 70, 60, 20]

fpr, tpr, thresholds = roc_curve(y_true, y_score)
print(auc(fpr, tpr))                  # 0.75
print(roc_auc_score(y_true, y_score)) # 0.75
```

**重要：** `auc(x, y)` 是對**已排序的曲線座標**求面積；`roc_auc_score(y_true, y_score)` 則直接接收真實標籤與原始分數。

#### 用單一 Feature 計算 AUC

```python
# 評估模型分數
roc_auc_score(y_val, y_pred_proba)

# 評估單一數值 Feature
roc_auc_score(df['converted'], df['age'])
```

若年齡越小越容易 `converted=1`，可反轉分數方向：

```python
roc_auc_score(df['converted'], -df['age'])
```

沒有同分時，反轉方向後的 AUC = `1 - 原 AUC`（有同分、以半分計算時也成立）。若用於特徵選擇，方向應只根據**訓練資料**決定。

### 5. 三者何時可能得出不同結論？

假設 **X 太低或太高時，y=1 的機率都較高（U 型關係）**：

- **Pearson Correlation**：可能接近 0（線性方向互相抵銷）。
- **MI**：可能大於 0（仍偵測到依賴關係）。
- **直接拿 X 計算 ROC AUC**：可能接近 0.5（單純依 X 大小排序無法分開兩類）。

因此，**AUC 接近 0.5 不代表 Feature 一定沒用**；它可能需要非線性轉換，或與其他特徵一起使用。

### 6. 實務使用建議

- **EDA 快速看方向** → Pearson Correlation。
- **想找非線性、複雜依賴** → Mutual Information。
- **想知道單一數值特徵能否直接排序正負類** → ROC AUC。
- **想知道加入某個特徵是否真的改善模型** → 在交叉驗證中比較「有／無該特徵」的模型表現。

> **一句話記憶：Correlation 看線性；MI 看資訊；ROC AUC 看排序。**
