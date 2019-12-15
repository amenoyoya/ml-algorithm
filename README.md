# 一から学ぶ機械学習アルゴリズム

## What's this?

Juliaを使い、機械学習の基礎的なアルゴリズムを実装するプロジェクト

### What's Julia?
- **JUlia(ジュリア)**
    - 汎用プログラミング言語水準から高度の計算科学や数値解析水準まで対処するよう設計された高水準言語かつ仕様記述言語、及び動的プログラミング言語
    - 標準ライブラリがJulia自身により作成されており、コア言語は極めてコンパクト
    - オブジェクトの構築または記述する型の種類が豊富にある
    - マルチディスパッチにより、引数の型の多くの組み合わせに対して関数の動作を定義することが可能
    - 異なる引数の型の効率的な特殊コードの自動生成が可能
    - C言語のような静的にコンパイルされた言語を使用しているかのような高いパフォーマンスを発揮する
    - フリーオープンソース（MITライセンス）
    - ユーザ定義の型は既存の型と同じくらい早くコンパクト
    - 非ベクトル化コード処理が早いため、パフォーマンス向上のためにコードをベクトル化する必要がない
    - 並列処理と分散計算ができるよう設計
    - 軽く「エコ」なスレッド（コルーチン）
    - 控えめでありながら処理能力が高いシステム
    - 簡潔かつ拡張可能な数値および他のデータ型のための変換と推進
    - 効率的なUnicodeへの対応（UTF-8を含む）
    - C関数を直接呼び出すことが可能（ラッパーや特別なAPIは不要）
    - 他のプロセスを管理するための処理能力が高いシェルに似た機能
    - Lispに似たマクロや他のメタプログラミング機能


***

## Setup

### Environment
- OS:
    - Ubuntu 18.04
    - Windows 10
- Jupyter Notebook: `4.4.0`
    - Anaconda: `4.5.11`
        - Python: `3.7.4`
- Julia: `1.3.0`

---

### Installation

#### Installation in Ubuntu
```bash
# install in home directory
$ cd ~

# download julia-1.3.0
$ wget -O - https://julialang-s3.julialang.org/bin/linux/x64/1.3/julia-1.3.0-linux-x86_64.tar.gz | tar zxvf -

# export path
## /usr/local/bin/ に julia 実行ファイルのシンボリックを作成
$ sudo ln -s ~/julia-1.3.0/bin/julia /usr/local/bin/julia

# confirm version
$ julia -v
julia version 1.3.0
```

#### Installation in Windows
管理者権限のPowerShellで以下を実行

```powershell
# chocolateyパッケージマネージャを入れていない場合は導入する
> Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# install julia
> choco install -y julia

# confirm version
> julia -v
julia version 1.3.0
```

---

### Julia でよく使うパッケージを導入
- HTTP:
    - HTTP通信を行うためのパッケージ
- DataFrames:
    - データフレームを扱うためのパッケージ

```bash
$ julia -e 'using Pkg; Pkg.add("HTTP"); Pkg.add("DataFrames")'
```

---

### Jupyter Notebook 導入
※ Anaconda導入済みと想定

```bash
# install jupyter notebook
$ conda install jupyter

# Jupyter Notebook 用のJuliaカーネルのインストール
$ julia -e 'using Pkg; Pkg.add("IJulia"); Pkg.build("IJulia")'

# jupyter notebook のカーネルを確認
$ jupyter kernelspec list
Available kernels:
  julia-1.3    /home/user/.local/share/jupyter/kernels/julia-1.3   # <- Juliaが使えるようになっている
  python3      /home/user/miniconda3/share/jupyter/kernels/python3

# Jupyter Notebook 起動
$ jupyter notebook

# => localhost:8888 で Jupyter Notebook 起動
## Juliaを使うには New > Julia 1.3.0 を選択する
```

---

### 機械学習用パッケージ導入
※ Anaconda導入済みと想定

```bash
# install ScikitLearn, matplotlib
$ conda install scikit-learn matplotlib

# install Julia PyCall package
$ julia -e 'using Pkg; Pkg.add("PyCall")'

# install Julia ScikitLearn bundled package
$ julia -e 'using Pkg; Pkg.add("ScikitLearn")'

# install Julia matplotlib bundled package
$ julia -e 'using Pkg; Pkg.add("PyPlot")'

# install Julia MachineLeaning Datasets package
$ julia -e 'using Pkg; Pkg.add("MLDatasets")'
```
