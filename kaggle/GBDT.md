# Kaggle 入門

## XGBoostモデルによる「タイタニック号の生存予測」

### Kernel
```python
import numpy as np # NumPy: 数値計算ライブラリ
import pandas as pd # Pandas: データフレームライブラリ

# 訓練用データの読み込み
train = pd.read_csv("../input/titanic/train.csv")

# 検証用データの読み込み
test = pd.read_csv("../input/titanic/test.csv")

# --- 特徴量の作成 ---

from sklearn.preprocessing import LabelEncoder

# 訓練用データを特徴量と目的変数に分ける
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 検証用データに Survived カラム（目的変数）はないため、そのままで良い
test_x = test.copy()

# PassengerId, Name, Ticket, Cabin カラムを削除
train_x = train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 各カテゴリ変数に Label Encoding を適用
for c in ['Sex', 'Embarked']:
    # 訓練用データに基づいてどう変換するか定める
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA')) # 欠損値は 'NA' として最適化する
    
    # 訓練・検証用データを最適化したLabelEncoderで変換
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# --- XGBoostモデルで分類モデルの最適化 ---
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# 最適化したモデルで検証データの生存予測
pred = model.predict_proba(test_x)[:, 1]

# 予測結果を二値変換
pred_label = np.where(pred > 0.5, 1, 0)

# 提出用ファイルの作成
sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred_label
})
sub.to_csv("submission_xgb.csv", index=False)
```

### Result
- 予測精度: 77.99 %
- ランキング: 5924 位

![xgboost_1.png](./img/xgboost_1.png)
