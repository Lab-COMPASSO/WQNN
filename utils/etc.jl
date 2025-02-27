using DataFrames, Query
using StatsBase
using Dates
using CSV


include("wnn_models.jl")
using .thermometer

function get_encoders(df)::Dict{String, thermometer.Thermometer}
    q = @from i in df begin
        @where i.mg > 0
        @select i
        @collect DataFrame
    end
    return Dict{String, thermometer.Thermometer}(
        "ghi"=>thermometer.fit!(thermometer.DistributiveEncoder(600, []), q[!,:mghi]),     
        "zenith"=>thermometer.fit!(thermometer.DistributiveEncoder(240, []), df[!,:zenith]),
        "azimuth"=>thermometer.fit!(thermometer.DistributiveEncoder(240, []), df[!,:azimuth]),
        "angle"=>thermometer.fit!(thermometer.LinearEncoder(0,0, 240), df[!,:mangle])
    )
end

function encode_func(ctx::DataFrameRow, encoders::Dict{String, thermometer.Thermometer})::BitArray
    a = BitArray([])
    append!(a, thermometer.encode(encoders["ghi"], ctx["ghi"]))
    append!(a, thermometer.encode(encoders["zenith"], ctx["zenith"]))
    append!(a, thermometer.encode(encoders["azimuth"], ctx["azimuth"]))
    append!(a, thermometer.encode(encoders["angle"], ctx["angle"]))
    
    return a
end