# Scalaによる機械学習実装

## What's this?

Scalaを使い、機械学習の基礎的なアルゴリズムを実装するプロジェクト

基本的には、Python実装である[ml-dlリポジトリ](https://github.com/amenoyoya/ml-dl)の[4.基礎理論](https://github.com/amenoyoya/ml-dl/4.基礎理論)の内容を反映している

***

## Setup

### Environment
- OS:
    - Windows 10 x64
- JavaVM:
    - OracleJDK: `12.0.1`
- Scala:
    - sbt: `2.12.7`

---

### Javaのインストール
- 現時点で最新の12系を[OpenJDK公式サイト](http://jdk.java.net/12/)からダウンロードしてくる
- `openjdk-12.0.1_windows-x64_bin.zip`を解凍
    - 以降、`C:\App\jdk-12`に解凍したと想定
- パスを通す
    - `Win + Pause/Break` => システム > システムの詳細 > 環境変数
        - PATH: `C:\App\jdk-12.0.1\bin`を追加
- コマンドプロンプトを起動し、Javaが使えるか確認
    ```bash
    # Javaバージョン確認
    > java -version
    openjdk version "12.0.1" 2019-04-16
    OpenJDK Runtime Environment (build 12.0.1+12)
    OpenJDK 64-Bit Server VM (build 12.0.1+12, mixed mode, sharing)

    # Javaコンパイラのバージョン確認
    > javac -version
    javac 12.0.1
    ```

---

### Scalaインストール
chocolateyを用いてsbtをインストールする

- 管理者権限でPowerShell起動
    ```powershell
    # chocolateyインストール
    > Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

    # バージョン確認
    > choco -v
    0.10.15

    # sbtインストール
    > choco install sbt
    ## => Do you want to run the script? というプロンプトに対しては「A」(All)と打ってOK
    ```
- 無事インストールされたら、`./00.perceptron`ディレクトリでPowerShellかコマンドプロンプトを起動
    ```bash
    # ScalaをREPL（Read Eval Print Loop）モードで起動
    > sbt console

    ## => 初回起動時は環境構築のため少し時間がかかる

     : (略)
    [info] Starting scala interpreter...
    Welcome to Scala 2.12.7 (OpenJDK 64-Bit Server VM, Java 12.0.1).
    Type in expressions for evaluation. Or try :help.
    ```
- Test run: パーセプトロンを実行してみる
    ```bash
    # 00.perceptron.scala で定義されている Perceptronオブジェクトの executeメソッドを実行
    scala> Perceptron.execute(List(1, 0), List(0.5, 0.5), 0.75)

    # 上記の実行結果として0が返ってくればテスト成功
    res0: Int = 0

    # (':quit' | ':q') コマンドでREPL終了
    scala> :q
    ```
