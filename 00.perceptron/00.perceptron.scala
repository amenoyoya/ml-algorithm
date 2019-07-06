// パーセプトロン
object Perceptron {
  // 信号を受け取り、重みと閾値から出力（0 | 1）を計算
  def execute(x: List[Int], w: List[Double], theta: Double): Int = {
    // [入力信号] * [重み] についてマッチング
    (x(0) * w(0) + x(1) * w(1)) match {
      // 閾値を超えているなら 1を返す
      case y if y > theta => 1
      case _ => 0
    }
  }
}
