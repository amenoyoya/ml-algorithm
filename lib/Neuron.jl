"""
NueralNetwork module

MIT License
Copyright (c) 2019 amenoyoya
"""

""" ニューラルネットワーク """

# ニューラルネットワーク構造体
mutable struct Network
    l::Int # ネットワーク層数
    b::Vector{Array{Float64,2}} # バイアス: [層1: Array{1x次層ニューロン数}, 層2: Array{1x次層ニューロン数}, ...]
    w::Vector{Array{Float64,2}} # 重み: [層1: Array{前層ニューロン数x次層ニューロン数}, 層2: Array{前層ニューロン数x次層ニューロン数}, ...]
end

# 順方向伝搬関数
## Network構造体, 層番号, 活性化関数, 層の入力信号 -> 層の出力信号 z
forward(network::Network, i::Int, h, x::Array{Float64,2})::Array{Float64,2} =
    # バッチ処理の場合、前層の行数が1以上になるため、バイアスの加算をループ加算（.+）に変更
    h(x * network.w[i] .+ network.b[i])

# 推論処理
## Network構造体, 中間層の活性化関数, 出力層の活性化関数, 入力信号, 層番号=1 -> 出力信号 y
predict(network::Network, h, σ, x::Array{Float64,2}, i::Int=1)::Array{Float64,2} = 
    i == network.l ? forward(network, i, σ, x) : predict(network, h, σ, forward(network, i, h, x), i + 1)


""" 活性化関数 """

# バッチ対応版シグモイド関数
sigmoid(a::Array{Float64,2})::Array{Float64,2} = reshape(vcat(
        [1 ./ (1 .+ exp.(-a[row, :])) for row in 1:size(a, 1)]...
    ), size(a, 1), size(a, 2)
)

# バッチ対応版ソフトマックス関数
softmax(a::Array{Float64,2})::Array{Float64,2} = begin
    y = []
    for row in 1:size(a, 1)
        c = maximum(a[row, :])
        exp_a = exp.(a[row, :] .- c) # オーバフロー対策
        push!(y, exp_a ./ sum(exp_a))
    end
    reshape(vcat(y...), size(a, 1), size(a, 2))
end


""" 損失関数 """

# 2乗和誤差
## Float64行列 予測値, Float64行列 正解値 -> Float64 誤差
mean_squared_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = 0.5 * sum((y - t).^2)

# 交差エントロピー誤差
## ln(0) = -Inf が発生するのを防ぐため、予測値に微小な値（10^-7）を加算して計算する
cross_entropy_error(y::Array{Float64,2}, t::Array{Float64,2})::Float64 = -sum(t .* log.(y .+ 1e-7))
