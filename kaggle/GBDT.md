# Kaggle 入門

## GBDTによる「タイタニック号の生存予測」

タイタニック号の生存予測問題のようにテーブルデータの分析においては、安定して高い精度が期待できるGBDTをファーストチョイスとするのが主流である

### GBDTとは
- 参考: https://www.acceluniverse.com/blog/developers/2019/12/gbdt.html

GBDTは、以下のような特徴を有する

- 精度が比較的高い
- 欠損値を扱える
- 不要な特徴量があっても精度が落ちにくい
- 汎用性が高い

![GDBT.png](https://www.acceluniverse.com/blog/developers/c6188796e9c3d19bca2bbf58a27ab6cae145fa71.png)

GBDTとは以下の手法を組み合わせたものである

- Gradient: 勾配降下法
    - 機械学習モデルの予測誤差を最小化するための手法
    - 参考: https://github.com/amenoyoya/ml-algorithm/blob/master/04_gradient.ipynb
- Boosting: アンサンブル手法の一種
    - アンサンブル手法とは多数決の原理を利用した学習手法である
        - 精度の低い学習器を複数組み合わせて精度を高くする手法
    - Boosting は、精度の低い学習器を順番に学習して組み合わせる手法である
- Decision Tree: 決定木
    - 決定木は木構造を用いて分類・回帰問題を解く手法で、段階的にデータを分析・分離することで目標値に関する推定結果を返す
    - ![decision-tree.png](https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.amazonaws.com%2F0%2F156509%2F04fd204e-af67-57eb-a7fa-8a1c241510f9.png?ixlib=rb-1.2.2&auto=format&gif-q=60&q=75&s=aefdea03346f31bf466a48a2f767db3a)


### Kernel
ここでは GBDT 実装の一つである XGBoost モデルを採用する

XGBoost は単純な決定木の代わりに Random Forests（決定木を弱学習器とするアンサンブル学習手法）を採用することで、汎化能力を向上させた GDBT 実装である

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
## 特徴量として使えない or 使いづらいカラムを削除して特徴量選択
train_x = train_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# 各カテゴリ変数に Label Encoding を適用
## LabelEncoder は与えられたラベルを最適な数値に自動的にフィッティングすることが可能
## => 手動で数値を割り振るより格段に楽
for c in ['Sex', 'Embarked']:
    # 欠損値は 'NA' とする
    ## 欠損値を補完しなくても使えるのがGBDTの良いところ
    x = train_x[c].fillna('NA')

    # 訓練用データに基づいてどう変換するか定める
    le = LabelEncoder()
    le.fit(x)
    
    # 訓練・検証用データを、最適化したLabelEncoderで数値データに変換
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

このように、モデルをロジスティック回帰から GBDT（XGBoost）に変えるだけでも、性能は大きく向上することがある

各学習モデルの長所・短所を知り、どういったデータに対してはどのようなモデルが適しているのか、よく理解することが重要である
