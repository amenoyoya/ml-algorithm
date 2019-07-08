// パーセプトロン
object Perceptron {
  // 信号を受け取り、[(<入力信号> * <重み>) + <バイアス>] から出力（0 | 1）を計算
  // Scala: 単純は配列なら List より Seq の方が良い（ランダムアクセスが速い）
  def execute(w: Seq[Double], b: Double)(x: Seq[Int]): Int = {
    // <入力信号> * <重み> + <バイアス> についてマッチング
    (x(0) * w(0) + x(1) * w(1) + b) match {
      // 0を超えているなら 1を返す
      case y if y > 0 => 1
      case _ => 0
    }
  }
}

// テスト
object Test {
  def main(): Int = {
    val AND = Perceptron(Seq(0.5, 0.5), -0.75)(_)
    println(AND(Seq(0, 0)))
  }
}