using DataFrames, Query
using StatsBase
using Dates
using CSV


include("irradiance.jl")
include("model.jl")
using .forget_model
using .thermometer

function get_encoders(df, n::Int)::Dict{String, thermometer.Thermometer}
    q = @from i in df begin
        @where i.mg > 0
        @select i
        @collect DataFrame
    end
    n_ = floor(Int,n*0.15)
    return Dict{String, thermometer.Thermometer}(
        "ghi"=>thermometer.fit!(thermometer.DistributiveEncoder(n-n_, []), q[!,:mghi]),     
        "zenith"=>thermometer.fit!(thermometer.DistributiveEncoder(n_, []), df[!,:zenith]),
        "azimuth"=>thermometer.fit!(thermometer.DistributiveEncoder(n_, []), df[!,:azimuth]),
        "angle"=>thermometer.fit!(thermometer.LinearEncoder(0,0, n_), df[!,:mangle])
       )
end


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

function run_episode(df, path, models::Array{forget_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, encoder_size::Int, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n_steps=$(n_steps)_encoder-size=$(encoder_size)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, n_steps, epsilon, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t_ < T_ - 1
        if t < T_
            nxtSt, reward, done, irrad = step(env, A[t])
            push!(S, nxtSt)
            push!(R, reward)

            dfr[nxtSt.i,"reward"] = reward
            dfr[nxtSt.i,"POA"] = irrad
            dfr[S[t].i,"angle"] = S[t].angle

            if done 
                T_ = t + 1
            else
                push!(A, getAction(env, S[t]))
            end
        end

        t_ = t - n_steps + 1
        if t_ > 1
            i = Int64(t_ + 1)
            i_max = Int64(min(t_ + n_steps, T_))
            G = sum([learning_rate^(i - t_ - 1) * R[i] for i in i:i_max])
            if t_ + n_steps < T_
                G = G + learning_rate^n_steps * forget_model.predict(env.Q[A[t_]], BinInput(env, S[t_]))
            end

            eps = env.learning_rate
            gamma = env.decay_rate
            
            forget_model.train!(
                env.Q[A[t_]],
                BinInput(env, S[t_]), 
                G
            ) 
        end
        t = t + 1      
    end

    if (index%25 == 0)|(index>(2500-25))
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end   


function run_episode(df, path, models::Array{forget_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n_steps=$(n_steps)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, n_steps, epsilon, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t_ < T_ - 1
        if t < T_
            nxtSt, reward, done, irrad = step(env, A[t])
            push!(S, nxtSt)
            push!(R, reward)

            dfr[nxtSt.i,"reward"] = reward
            dfr[nxtSt.i,"POA"] = irrad
            dfr[S[t].i,"angle"] = S[t].angle

            if done 
                T_ = t + 1
            else
                push!(A, getAction(env, S[t]))
            end
        end

        t_ = t - n_steps + 1
        if t_ > 1
            i = Int64(t_ + 1)
            i_max = Int64(min(t_ + n_steps, T_))
            G = sum([learning_rate^(i - t_ - 1) * R[i] for i in i:i_max])
            if t_ + n_steps < T_
                G = G + learning_rate^n_steps * forget_model.predict(env.Q[A[t_]], BinInput(env, S[t_]))
            end

            eps = env.learning_rate
            gamma = env.decay_rate
            
            forget_model.train!(
                env.Q[A[t_]],
                BinInput(env, S[t_]), 
                G
            ) 
        end
        t = t + 1      
    end

    if (index%25 == 0)
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end 

function run_episode_eval(df, path, models::Array{forget_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n_steps=$(n_steps)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, n_steps, epsilon, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t_ < T_ - 1
        if t < T_
            nxtSt, reward, done, irrad = step(env, A[t])
            push!(S, nxtSt)
            push!(R, reward)

            dfr[nxtSt.i,"reward"] = reward
            dfr[nxtSt.i,"POA"] = irrad
            dfr[S[t].i,"angle"] = S[t].angle

            if done 
                T_ = t + 1
            else
                push!(A, getAction(env, S[t]))
            end
        end

        t_ = t - n_steps + 1
        if t_ > 1
            i = Int64(t_ + 1)
            i_max = Int64(min(t_ + n_steps, T_))
            G = sum([learning_rate^(i - t_ - 1) * R[i] for i in i:i_max])
            if t_ + n_steps < T_
                G = G + learning_rate^n_steps * forget_model.predict(env.Q[A[t_]], BinInput(env, S[t_]))
            end

            eps = env.learning_rate
            gamma = env.decay_rate
            
            forget_model.train!(
                env.Q[A[t_]],
                BinInput(env, S[t_]), 
                G
            ) 
        end
        t = t + 1      
    end

    CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
end   


struct State
    i :: Int
    angle :: Float64
end

mutable struct Enviroment
    df::DataFrame
    Q::Array{forget_model.Model}
    encoders::Dict{String, thermometer.Thermometer}
    state::State
    epsilon::Float64
    n_steps::Float64
    learning_rate::Float64
    decay_rate::Float64
end


function get_enviroment(df, models::Array{forget_model.Model}, encoders::Dict{String, thermometer.Thermometer}, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64)
    nans = [NaN for i in 1:nrow(df)]
    df.angle = nans
    df.POA = nans
    df.reward = nans
        
    return Enviroment(df, 
        models,
        encoders,
        State(1,0),
        epsilon,
        n_steps,
        learning_rate,
        decay_rate
    )
end


function get_enviroment(df, tuple_size::Int, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, nactions::Int)
    nans = [NaN for i in 1:nrow(df)]
    df.angle = nans
    df.POA = nans
    df.reward = nans
    encoders = get_encoders(df)
    
    input_size = length(encode_func(df[1,:],encoders))
        
    return Enviroment(df, 
        [generate_forget_model.Model(input_size, tuple_size, learning_rate) for i in 1:nactions],
        get_encoders(df),
        State(1,0),
        epsilon,
        learning_rate,
        learning_rate,
        decay_rate
    )
end


function BinInput(env, state)
        # print(state, file=sys.stderr)
    ctx = env.df[state.i,:]
    ctx["angle"] = state.angle

    return encode_func(ctx[["ghi","zenith","azimuth","angle"]], env.encoders)
end

function Q_predict(env, state, action)
    Qfunction = env.Q[action]    
    return forget_model.predict(Qfunction, BinInput(env, state))
end
    
function get_context(env::Enviroment, state::State)
    return env.df[state.i, :]
end
    
function getResult(env, state)
    ctx = env.df[state.i,:]
    irrad = get_total_irradiance(state.angle, 90, ctx["zenith"], ctx["azimuth"], ctx["dni"], ctx["dhi"], ctx["ghi"], 0.2)
    dt = DateTime(ctx.ts, "yyyy-mm-dd HH:MM:SS")
    bonus = 0
    if ((hour(dt) <= 4)||(hour(dt) >= 20)) && abs(state.angle) > 10
        bonus = -50
    end

    return (irrad + bonus)^3, irrad
end
    

function nextState(env, action)
    state = env.state
    ctx = env.df[state.i,:]

    dt = DateTime(ctx.ts, "yyyy-mm-dd HH:MM:SS")

    if (hour(dt) <= 4)||(hour(dt) >= 20)
        env.state = State(state.i + 1, 0)
    elseif action == 1
        env.state = State(state.i + 1, minimum([60., state.angle + 5]))
    elseif action == 2
        env.state = State(state.i + 1, maximum([-60., state.angle  - 5]))
    elseif action == 3
        env.state = State(state.i + 1, minimum([60., state.angle  + 10]))
    elseif action == 4
        env.state = State(state.i + 1, maximum([-60., state.angle  - 10]))
    else
        env.state = State(state.i + 1, state.angle)
    end
    return env.state
end   
    
    
function step(env, action)
    stt = nextState(env, action)
    reward, irrad = getResult(env, stt)
    done = stt.i > nrow(env.df)    
    
    return stt, reward, done, irrad
end        

    
function EGreedyPolicy(Q_table, epsilon)
    if rand() < epsilon
        return argmax(Q_table)
    end
    return sample(1:length(Q_table))
end
        
function getBestAction(env, state)
        return argmax([Q_predict(env, state, action) for action in 1:length(env.Q)])
end
        
function getAction(env, state)
    Q_table = [Q_predict(env, state, action) for action in 1:length(env.Q)]
    return EGreedyPolicy(Q_table, env.epsilon)
end