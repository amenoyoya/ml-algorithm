"""
NueralNetwork Functions module

MIT License
Copyright (c) 2019 amenoyoya
"""

"""
活性化関数: 入力値に何らかの処理を施し出力する
    入力次元 = 出力次元となる
"""

# 恒等関数
identity_function(x::Float64)::Float64 = x

# ステップ関数
step_function(x::Float64)::Float64 = x > 0 ? x : 0

# シグモイド関数
sigmoid(x::Array{Float64,1})::Array{Float64,1} = 1 ./ (1 .+ exp.(-x))    

# シグモイド関数（バッチ対応）
sigmoid(x::Array{Float64,2})::Array{Float64,2} = hcat([sigmoid(x[row, :]) for row in 1:size(x, 1)]...)'

# ReLU関数
relu(x::Array{Float64,1})::Array{Float64,1} = maximum.(0.0, x)

# ReLU関数（バッチ対応）
relu(x::Array{Float64,2})::Array{Float64,2} = hcat([relu(x[row, :]) for row in 1:size(x, 1)]...)'

# ソフトマックス関数
softmax(x::Array{Float64,1})::Array{Float64,1} = begin
    x = x .- maximum(x) # オーバーフロー対策
    exp.(x) ./ sum(exp.(x))
end

# ソフトマックス関数（バッチ対応）
softmax(x::Array{Float64,2})::Array{Float64,2} = hcat([softmax(x[row, :]) for row in 1:size(x, 1)]...)'


"""
損失関数: 得た予測値と正解値の誤差をFloat64型で出力
    予測次元 = 正解次元 ≠ 出力次元 となる
"""

# 二乗和誤差
## 予測値, 正解値 -> 誤差
mean_squared_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = 0.5 * sum((y .- t).^2)

# 交差エントロピー誤差
## 予測値, 正解値 -> 誤差
cross_entropy_error(y::Array{Float64,1}, t::Array{Float64,1})::Float64 = begin
    # 教師データ: one-hot-vectorを想定 => 正解ラベルのindexのみ計算
    val, index = findmax(t) # 正解ラベル = 最大値(=1.0)のindex
    # ln(0) = -Inf が発生するのを防ぐため、予測値に微小な値（10^-7）を加算して計算する
    -log(y[index] .+ 1e-7)
end

# 交差エントロピー誤差（バッチ対応）
## 予測値, 正解値 -> 誤差
cross_entropy_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = begin
    batch_size = size(y, 1)
    sum([cross_entropy_error(y[row, :], t[row, :]) for row in 1:batch_size]) / batch_size
end
