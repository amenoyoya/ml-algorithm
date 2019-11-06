"""
NueralNetwork module

MIT License
Copyright (c) 2019 amenoyoya
"""

# 数値微分関数
numeric_diff(f, x) = begin
    h = 1e-4 # 10^(-4)
    (f(x + h) - f(x - h)) / 2h # 中心差分から微分値を計算
end

# 数値微分による勾配計算
## 関数 f(Float64配列), Float64配列 x -> Float64配列 勾配
numeric_gradient(f, x::Array{Float64,1})::Array{Float64,1} = begin
    h = 1e-4 # 10^(-4)
    grad = Array{Float64, 1}(undef, length(x)) # xと同じ長さの配列 [undef, undef, ...] を生成
    # 各変数ごとの数値微分を配列にまとめる
    for i in 1:length(x)
        # 指定indexの変数に対する中心差分を求める
        org = x[i]
        x[i] = org + h
        f1 = f(x) # f([..., x[i] + h, ...])
        x[i] = org - h
        f2 = f(x) # f([..., x[i] - h, ...])
        grad[i] = (f1 - f2) / 2h # i番目の変数に対する数値微分
        x[i] = org # x[i]の値をもとに戻す
    end
    return grad
end

# 勾配降下法
## 損失関数 f, ベクトル x, 学習率 lr, 試行回数 ste_num -> 最適化ベクトル
gradient_descent(f, x::Array{Float64,1}, lr::Float64=0.01, step_num::Int=100)::Array{Float64,1} = begin
    step_num > 0 ? gradient_descent(f, x - lr * numeric_gradient(f, x), lr, step_num - 1) : x
end
