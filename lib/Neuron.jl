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
