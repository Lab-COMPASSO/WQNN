using DataFrames, Query
using StatsBase
using Dates
using CSV


include("irradiance.jl")
include("model.jl")
using  .deepQ_model


function run_episode(df::DataFrame, path::String, model::deepQ_model.Model, hidden_layer_size::Int64, epsilon::Float32, learning_rate::Float32, decay_rate::Float32, index::Int)
    mypath = joinpath(path, "epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_hidden-size=$(hidden_layer_size)")
    mkpath(mypath)
    filename = joinpath(mypath, "checkpoint_$(index).csv")

    env = get_enviroment(df, model, epsilon, learning_rate, decay_rate)
    done = false
    state = env.state
    dfr = copy(env.df)

    while done == false && state.i < nrow(env.df)-1
        ctx = get_context(env, state)
        action = getAction(env, state)
        
        ctx.angle = state.angle
        nxtSt, reward, done, irrad = step(env, action)

        Q  = Q_array(env, state)
        nextAction = argmax(Q)
        Q_ = Q_array(env, nxtSt)

        

        eps = env.learning_rate
        gamma = env.decay_rate

        Q[action] = (1 - eps) * Q[action] + eps * (reward  +  gamma*Q_[nextAction])

        dfr[nxtSt.i,"reward"] = reward
        dfr[nxtSt.i,"POA"] = irrad
        dfr[state.i,"angle"] = state.angle

        
        deepQ_model.train!(env.Q, collect(Float32, ctx[["ghi","zenith","azimuth","angle"]]), Q)

        

        state = nxtSt

       
    end

    if (index%25 == 0)|(index>(5000-25))
        println(mypath, index)
        
        CSV.write(filename, filter(row -> ! isnan(row.POA), dfr))
    end
end   



struct State
    i :: Int
    angle :: Float32
end

mutable struct Enviroment
    df::DataFrame
    Q::deepQ_model.Model
    state::State
    epsilon::Float32
    learning_rate::Float32
    decay_rate::Float32
end


function get_enviroment(df::DataFrame, model::deepQ_model.Model, epsilon::Float32, learning_rate::Float32, decay_rate::Float32)
    nans = [NaN for i in 1:nrow(df)]
    df[!, :angle] .= 0
    df.POA = nans
    df.reward = nans
        
    return Enviroment(df, 
        model,
        State(1,0),
        epsilon,
        learning_rate,
        decay_rate
    )
end

function Q_array(env::Enviroment, state::State)
    Qfunction = env.Q
    ctx = env.df[state.i,:]
    return deepQ_model.predict(Qfunction, collect(Float32, ctx[["ghi","zenith","azimuth","angle"]]))
end

function Q_predict(env::Enviroment, state::State, action::Int64)
    Qact = Q_array(env, state)
    return Qact[action]
end
    
function get_context(env::Enviroment, state::State)
    return env.df[state.i, :]
end
    
function getResult(env, state)
    ctx = env.df[state.i,:]
    irrad = get_total_irradiance(convert(Float64,state.angle), 90, ctx["zenith"], ctx["azimuth"], ctx["dni"], ctx["dhi"], ctx["ghi"], 0.2)
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
        



function getBestAction(env::Enviroment, state::State)
    return argmax(Q_array(env, state))
end
        
function getAction(env::Enviroment, state::State)
    Q_table = Q_array(env, state)
    return EGreedyPolicy(Q_table, env.epsilon)
end