using DataFrames, Query
using StatsBase
using Dates
using CSV


include("irradiance.jl")
include("WQNN.jl")

function get_encoders(df)::Dict{String, Thermometer}
    q = @from i in df begin
        @where i.mg > 0
        @select i
        @collect DataFrame
    end
    return Dict{String, Thermometer}(
        "ghi"=>fit!(DistributiveEncoder(600, []), q[!,:mghi]),     
        "zenith"=>fit!(DistributiveEncoder(240, []), df[!,:zenith]),
        "azimuth"=>fit!(DistributiveEncoder(240, []), df[!,:azimuth]),
        "angle"=>fit!(LinearEncoder(0,0, 240), df[!,:mangle])
       )
end

function encode_func(ctx::DataFrameRow, encoders::Dict{String, Thermometer})::BitArray
    a = BitArray([])
    append!(a, encode(encoders["ghi"], ctx["ghi"]))
    append!(a, encode(encoders["zenith"], ctx["zenith"]))
    append!(a, encode(encoders["azimuth"], ctx["azimuth"]))
    append!(a, encode(encoders["angle"], ctx["angle"]))
    
    return a
end


function run_episode(df, path, models::Array{Model}, encoders::Dict{String, Thermometer}, tuple_size::Int, forget_factor::Float64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, forget_factor, epsilon, learning_rate, decay_rate)
    done = false
    state = env.state
    dfr = copy(env.df)

    while done == false && state.i < nrow(env.df)-1

        action = getAction(env, state)
        nxtSt, reward, done, irrad = step(env, action)

        Q  = Q_predict(env, state, action)
        nextAction = getBestAction(env, nxtSt)
        Q_ =Q_predict(env, nxtSt, nextAction)

        eps = env.learning_rate
        gamma = env.decay_rate
        train!(env.Q[action],BinInput(env, state), (1 - eps) * Q + eps * (reward  +  gamma*Q_))

        dfr[nxtSt.i,"reward"] = reward
        dfr[nxtSt.i,"POA"] = irrad
        dfr[state.i,"angle"] = state.angle

        state = nxtSt

       
    end

    if (index%25 == 0)
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end   

function run_episode_eval(df, path, models::Array{Model}, encoders::Dict{String, Thermometer}, tuple_size::Int, forget_factor::Float64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, models, encoders, forget_factor, epsilon, learning_rate, decay_rate)
    done = false
    state = env.state
    dfr = copy(env.df)

    while done == false && state.i < nrow(env.df)-1

        action = getAction(env, state)
        nxtSt, reward, done, irrad = step(env, action)

        Q  = Q_predict(env, state, action)
        nextAction = getBestAction(env, nxtSt)
        Q_ =Q_predict(env, nxtSt, nextAction)

        eps = env.learning_rate
        gamma = env.decay_rate
        train!(env.Q[action],BinInput(env, state), (1 - eps) * Q + eps * (reward  +  gamma*Q_))

        dfr[nxtSt.i,"reward"] = reward
        dfr[nxtSt.i,"POA"] = irrad
        dfr[state.i,"angle"] = state.angle

        state = nxtSt

       
    end

    println(mypath, index)
    CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
end 


function run_individual(df, path, tuple_size::Int, epsilon::Float64, forget_factor::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, tuple_size, epsilon, forget_factor, learning_rate, decay_rate, 5)
    done = false
    state = env.state
    dfr = copy(env.df)
    println(mypath)
    while done == false && state.i < nrow(env.df)-1

        action = getAction(env, state)
        nxtSt, reward, done, irrad = step(env, action)

        Q  = Q_predict(env, state, action)
        nextAction = getBestAction(env, nxtSt)
        Q_ =Q_predict(env, nxtSt, nextAction)

        eps = env.learning_rate
        gamma = env.decay_rate
        train!(env.Q[action],BinInput(env, state), (1 - eps) * Q + eps * (reward  +  gamma*Q_))

        dfr[nxtSt.i,"reward"] = reward
        dfr[nxtSt.i,"POA"] = irrad
        dfr[state.i,"angle"] = state.angle

        state = nxtSt

        if nxtSt.i%15000 == 0
            println(mypath, nxtSt, env.epsilon)
            
            CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
        end
    end
    CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
end   



struct State
    i :: Int
    angle :: Float64
end

mutable struct Enviroment
    df::DataFrame
    Q::Array{Model}
    encoders::Dict{String, Thermometer}
    state::State
    epsilon::Float64
    forget_factor::Float64
    learning_rate::Float64
    decay_rate::Float64
end


function get_enviroment(df, models::Array{Model}, encoders::Dict{String, Thermometer}, forget_factor::Float64,epsilon::Float64, learning_rate::Float64, decay_rate::Float64)
    nans = [NaN for i in 1:nrow(df)]
    df.angle = nans
    df.POA = nans
    df.reward = nans
        
    return Enviroment(df, 
        models,
        encoders,
        State(1,0),
        epsilon,
        forget_factor,
        learning_rate,
        decay_rate
    )
end


function get_enviroment(df, tuple_size::Int, epsilon::Float64, forget_factor::Float64, learning_rate::Float64, decay_rate::Float64, nactions::Int)
    nans = [NaN for i in 1:nrow(df)]
    df.angle = nans
    df.POA = nans
    df.reward = nans
    encoders = get_encoders(df)
    
    input_size = length(encode_func(df[1,:],encoders))
        
    return Enviroment(df, 
        [generate_Model(input_size, tuple_size, forget_factor) for i in 1:nactions],
        get_encoders(df),
        State(1,0),
        epsilon,
        forget_factor,
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
    return predict(Qfunction, BinInput(env, state))
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
        bonus = -500
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