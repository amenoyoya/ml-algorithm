# Kaggle 入門

## Kaggle とは

- 機械学習・データサイエンスに携わっている人間が世界中から集まっているコミニティー
- 企業や政府などの組織と、データ分析のプロであるデータサイエンティスト/機械学習エンジニアを繋げるプラットフォームとなっている
- 単純なマッチングではなく、「Competetion（コンペ）」がKaggleの目玉の一つ
    - Competition（コンペ）は、企業や政府がコンペ形式（競争形式）で課題を提示し、賞金と引き換えに最も制度の高い分析モデルを買い取る仕組み
    - 開催されるコンペは様々で、アメリカ国土安全保障省による空港のセキュリティースクリーニングの認識アルゴリズムの競争や、日本のメルカリによる販売者への自動価格提案アルゴリズムなどが開催されている
    - Kagglerは無料でこれらのコンペに参加が可能で、企業から提供されているトレーニング用のデータセット（またそれに付随する様々なデータ）を利用して、モデルの訓練およびテストセットの評価ができる
- 初心者向けの一つの機能として「Kernels（カーネル）」というものもある
    - カーネルでは、各データセットに対して他のユーザーが構築した予測モデルのコードや説明が公開されている
    - 初心者にも優しく説明がされているカーネルも多数あり、カーネルを眺めているだけでも勉強になることが多い
    - カーネルはブラウザ上で実行可能な機械学習実行環境で、一般的なノートパソコン以上の性能の環境を無料で自由に使うことが可能
- 「Discussion（ディスカッション）」では、世界中のデータサイエンティスト・機械学習実装者とのコミュニケーションも可能

***

## Kernel を使ってみる

### Kernel を新規作成する
Kernel を新規作成する場合は、マイページの「Kernel」タブから「Create New Notebook」を選ぶ

![kaggle-kernel.png](./img/kaggle-kernel.png)

2020年1月時点で、Kernel として使用可能な言語は **Python** と **R** である

また、**Notebook（Jupyter Notebook）形式** か **Script形式** を選択することができる

![kaggle-kernel-new.png](./img/kaggle-kernel-new.png)

基本的には、インタラクティブに実行できる Notebook形式 がオススメである

***

## Kaggle コンペへの参加

- 参考:
    - https://qiita.com/upura/items/3c10ff6fed4e7c3d70f0
    - https://qiita.com/suzumi/items/8ce18bc90c942663d1e6

### タイタニック号の生存予測コンペ
[タイタニック号の生存予測](https://www.kaggle.com/c/titanic) は Kaggle におけるチュートリアル的なコンペである

まずは、このコンペに参加してみる

コンペページの「Join Competition」を押すと、ルールに同意するかどうかのダイアログが出てくるため「I Understand and Accept」をクリックする

![kaggle-compe-join.png](./img/kaggle-compe-join.png)

![kaggle-compe-accept.png](./img/kaggle-compe-accept.png)

コンペに参加できたら、「Notebook」タブから「New Notebook」を押す

すると、コンペ用の Kernel を作成できるため、使用言語と形式を選択する（今回は **Python** / **Notebook形式** とした）

![kaggle-compe-new.png](./img/kaggle-compe-new.png)

Notebook形式の Kernel を作ると以下のような画面になるはずである

![kaggle-kernel-notebook.png](./img/kaggle-kernel-notebook.png)

右側の「Settings」メニューから、Kernel の公開状態や使用言語、GPUなどを変更することができる

なお、GPUを使いたい場合は電話番号認証が必要である

### 電話番号認証
GPUが使えないと遅すぎて使い物にならないため、電話番号認証をしておく

「Requires phone validation」をクリックし、電話番号を入力する

![kaggle-phone-validation.png](./img/kaggle-phone-validation.png)

SMSでコード（数字）が送られてくるため、それを入力して「Verify」する

![kaggle-code-verify.png](./img/kaggle-code-verify.png)

***

## タイタニックの生存予測

実際に Kernel を用いて機械学習を行う

とりあえずは、u++ さんのコード（ロジスティック回帰による分類最適化）をそのまま実行してみる

```python
import numpy as np # NumPy: 数値計算ライブラリ
import pandas as pd # Pandas: データフレームライブラリ

# 訓練用データの読み込み
train = pd.read_csv("../input/titanic/train.csv")

# 検証用データの読み込み
test = pd.read_csv("../input/titanic/test.csv")

# --- 特徴量エンジニアリング ---

# 訓練・検証用データを合成
data = pd.concat([train, test], sort=False)

# Sex: male => 0, female => 1
data['Sex'].replace(['male','female'], [0, 1], inplace=True)

# Embarked: 欠損値を S で埋め、S => 0, C => 1, Q => 2 に変換 
data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Fare: 欠損データを平均値で埋める
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

# Age: 欠損データを (平均値±標準偏差) で埋める
age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

# 今回、特徴量として Name, PassengerId, SibSp, Parch, Ticket, Cabin は使わないことにする
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train_data = data[:len(train)]
test_data = data[len(train):]

# 教師データ: 訓練用データの `Survived` (生存したか否か) カラム
y_train = train_data['Survived']

# 特徴量データ: 訓練用データから `Surviced` カラムを抜いたもの
X_train = train_data.drop('Survived', axis=1)
X_test = test_data.drop('Survived', axis=1)

# --- ロジスティック回帰で分類ソルバ最適化 ---
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)

# 最適化したソルバで検証データの生存予測
y_pred = clf.predict(X_test)

# コンペ提出用データの読み込み
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
sub = gender_submission

# 最適化ソルバーの予測結果を `Survived` カラムにマッピング
sub['Survived'] = list(map(int, y_pred))

# 課題提出
sub.to_csv("submission.csv", index=False)
```

実行して問題なく動くようであれば、右上の「COMMIT」ボタンを押して、コミットする（コードを実際のデータに対して実行し、ソースコードと結果を保存する）

データ量が多いため、コミットには結構な時間がかかる
