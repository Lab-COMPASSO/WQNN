using DataFrames, Query
using StatsBase
using Dates
using CSV


include("irradiance.jl")
include("model.jl")
using  .policy_model
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


function run_episode(df::DataFrame, path::String, models::Array{policy_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t < T_
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
        t = t + 1      
    end

    for h in 1:(T_ - 1)
        G = sum([(env.decay_rate^(k - h -1))*R[k] for k in (h + 1):T_])
        grad = env.learning_rate * G * grad_ln_prob(env, S[h], A[h])
        
        
        for a in 1:length(env.fmodels)
            g = grad[a]
            if isnan(g)
                g = 0.
            end
            policy_model.train!(env.fmodels[a], BinInput(env, S[h]), g)
        end
    end



    if (index%25 == 0)
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end   


function run_episode_eval(df::DataFrame, path::String, models::Array{policy_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t < T_
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
        t = t + 1      
    end

    for h in 1:(T_ - 1)
        G = sum([(env.decay_rate^(k - h -1))*R[k] for k in (h + 1):T_])
        grad = env.learning_rate * G * grad_ln_prob(env, S[h], A[h])
        
        
        for a in 1:length(env.fmodels)
            g = grad[a]
            if isnan(g)
                g = 0.
            end
            policy_model.train!(env.fmodels[a], BinInput(env, S[h]), g)
        end
    end



    CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
end   

function run_episode(df::DataFrame, path::String, models::Array{policy_model.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, learning_rate::Float64, decay_rate::Float64, encoder_size, index::Int)
    mypath = joinpath(path, "learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)_encoder-size=$(encoder_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, learning_rate, decay_rate)
    done = false
    S = [env.state]
    R  = [0.0]
    
    dfr = copy(env.df)
    t = 1
    t_ = -Inf
    T_ = nrow(df)

    A = [getAction(env, S[t])]
    while t < T_
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
        t = t + 1      
    end

    for h in 1:(T_ - 1)
        G = sum([(env.decay_rate^(k - h -1))*R[k] for k in (h + 1):T_])
        grad = env.learning_rate * G * grad_ln_prob(env, S[h], A[h])
        
        
        for a in 1:length(env.fmodels)
            g = grad[a]
            if isnan(g)
                g = 0.
            end
            policy_model.train!(env.fmodels[a], BinInput(env, S[h]), g)
        end
    end



    if (index%25 == 0)|(index>(5000-25))
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end   


struct State
    i :: Int
    angle :: Float64
end

mutable struct Enviroment
    df::DataFrame
    fmodels::Array{policy_model.Model}
    encoders::Dict{String, thermometer.Thermometer}
    state::State
    learning_rate::Float64
    decay_rate::Float64
end

function get_enviroment(df::DataFrame, models::Array{policy_model.Model}, encoders::Dict{String, thermometer.Thermometer}, learning_rate::Float64, decay_rate::Float64)::Enviroment
    nans = [NaN for i in 1:nrow(df)]
    df.angle = nans
    df.POA = nans
    df.reward = nans
        
    
    return Enviroment(df, 
        models,
        encoders::Dict{String, thermometer.Thermometer},
        State(1,0),
        learning_rate,
        decay_rate
    )
end


function BinInput(env::Enviroment, state::State)::BitArray
        # print(state, file=sys.stderr)
    ctx = env.df[state.i,:]
    ctx["angle"] = state.angle

    return encode_func(ctx[["ghi","zenith","azimuth","angle"]], env.encoders)
end
    
function get_context(env::Enviroment, state::State)
    return env.df[state.i, :]
end
    
function getResult(env::Enviroment, state::State)
    ctx = env.df[state.i,:]
    irrad = get_total_irradiance(state.angle, 90, ctx["zenith"], ctx["azimuth"], ctx["dni"], ctx["dhi"], ctx["ghi"], 0.2)
    dt = DateTime(ctx.ts, "yyyy-mm-dd HH:MM:SS")
    bonus = 0
    if ((hour(dt) <= 4)||(hour(dt) >= 20)) && abs(state.angle) > 10
        bonus = -50
    end

    return (irrad + bonus)^3/(1200^3), irrad
end
    

function nextState(env::Enviroment, action::Int64)::State
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
    
    
function step(env::Enviroment, action::Int)
    stt = nextState(env, action)
    reward, irrad = getResult(env, stt)
    done = stt.i > nrow(env.df)    
    
    return stt, reward, done, irrad
end        

function grad_ln_prob(env::Enviroment, state::State, action::Int64)
    probs = PolicyProbs(env, state)
    # k = k_distance(env, state)
    n_actions = length(env.fmodels)

    return [a != action ? -probs[a] : (1-probs[a]) for a in 1:n_actions]
end

function k_distance(env::Enviroment, state::State)
    n_actions = length(env.fmodels)
    return [policy_model.get_k(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
end


function PolicyProbs(env::Enviroment, state::State)
    n_actions = length(env.fmodels)
    fs = [policy_model.get_f(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
    exp_f = [exp(fs[action]) for action in 1:n_actions]
    s = sum(exp_f)
        
    if s > 0
        return [exp_f[a]/s for a in 1:n_actions]
    else
        return [1/n_actions for a in 1:n_actions]
    end
end
    
function getAction(env::Enviroment, state::State)
    p = PolicyProbs(env, state)
    try
        probs = Weights(p)
        return sample(1:length(env.fmodels), probs)
    catch y
        # warn("Exception: ", y) # What to do on error.
        println(y)
        println([get_f(env.fmodels[action], BinInput(env, state)) for action in 1:length(env.fmodels)])
        println(p)
    end
end