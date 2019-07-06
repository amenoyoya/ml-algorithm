/*
  Scalaのバージョンを指定
  バージョンは `sbt console` で REPLを起動した時に表示される
*/
scalaVersion := "2.12.7"

/*
  コンパイルオプションを追加
  - deprecation: 今後廃止予定のAPIを使用する
  - feature: 明示的に仕様を宣言する必要のある機能を使用する
  - unchecked: パターンマッチが有効に機能しない場合に指定
  - Xlint: 推奨する書き方などの情報を出力
*/
scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Xlint")
