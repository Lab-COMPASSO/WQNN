include("physics.jl")
using .physics
include("wnn_models.jl")

function calc_reward(state, context)
    irrad = physics.get_total_irradiance(state.angle, 90, context["zenith"], context["azimuth"], context["dni"], context["dhi"], context["ghi"], 0.2)
    dt = DateTime(ctx.ts, "yyyy-mm-dd HH:MM:SS")
    bonus = 0
    if ((hour(dt) <= 4)||(hour(dt) >= 20)) && abs(state.angle) > 10
        bonus = -500
    end

    return (irrad + bonus)^3, irrad
    
end

module SARSA_Agent
    using DataFrames
    using .WQNN
    
    function run_episode(df, path, models::Array{WQNN.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64, index::Int)
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
                    G = G + learning_rate^n_steps * WQNN.predict(env.Q[A[t_]], BinInput(env, S[t_]))
                end

                eps = env.learning_rate
                gamma = env.decay_rate
                
                WQNN.train!(
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

    struct State
        i :: Int
        angle :: Float64
    end

    mutable struct Enviroment
        df::DataFrame
        Q::Array{WQNN.Model}
        encoders::Dict{String, thermometer.Thermometer}
        state::State
        epsilon::Float64
        n_steps::Float64
        learning_rate::Float64
        decay_rate::Float64
    end


    function get_enviroment(df, models::Array{WQNN.Model}, encoders::Dict{String, thermometer.Thermometer}, n_steps::Int64, epsilon::Float64, learning_rate::Float64, decay_rate::Float64)
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

    function BinInput(env, state)
        ctx = env.df[state.i,:]
        ctx["angle"] = state.angle

        return encode_func(ctx[["ghi","zenith","azimuth","angle"]], env.encoders)
    end

    function Q_predict(env, state, action)
        Qfunction = env.Q[action]    
        return WQNN.predict(Qfunction, BinInput(env, state))
    end
        
    function getResult(env::Enviroment, state::State)
        ctx = env.df[state.i,:]
        return calc_reward(state, ctx)
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
            
    function getAction(env, state)
        Q_table = [Q_predict(env, state, action) for action in 1:length(env.Q)]
        return EGreedyPolicy(Q_table, env.epsilon)
    end
end

module REINFORCE_Agent  
    using DataFrames
    
    function run_episode(df, path, models::Array{GDWPNN.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, learning_rate::Float64, decay_rate::Float64, index::Int)
        mypath = joinpath(path, "learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)")
        mkpath(mypath)
        filename = joinpath(mypath, "checkpoint_$(index).csv")

        env = get_enviroment(df, models, encoders, learning_rate, decay_rate, length(models))
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
                GDWPNN.train!(env.fmodels[a], BinInput(env, S[h]), g)
            end
        end

        if (index%25 == 0)|(index>(2500-25))
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
        fmodels::Array{GDWPNN.Model}
        encoders::Dict{String, thermometer.Thermometer}
        state::State
        learning_rate::Float64
        decay_rate::Float64
    end

    function get_enviroment(df, models::Array{GDWPNN.Model}, encoders::Dict{String, thermometer.Thermometer}, learning_rate::Float64, decay_rate::Float64, nactions::Int)
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


    function BinInput(env, state)
            # print(state, file=sys.stderr)
        ctx = env.df[state.i,:]
        ctx["angle"] = state.angle

        return encode_func(ctx[["ghi","zenith","azimuth","angle"]], env.encoders)
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

        return (irrad + bonus)^3/(1200^3), irrad
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

    function grad_ln_prob(env::Enviroment, state::State, action::Int64)
        probs = PolicyProbs(env, state)
        # k = k_distance(env, state)
        n_actions = length(env.fmodels)

        return [a != action ? -probs[a] : (1-probs[a]) for a in 1:n_actions]
    end

    function k_distance(env::Enviroment, state::State)
        n_actions = length(env.fmodels)
        return [GDWPNN.get_k(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
    end


    function PolicyProbs(env::Enviroment, state::State)
        n_actions = length(env.fmodels)
        fs = [GDWPNN.get_f(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
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
end

module AC_Agent
    using DataFrames
    
    function run_episode(df, path, vmodel::WQNN.Model, models::Array{GDWPNN.Model}, encoders::Dict{String, thermometer.Thermometer}, tuple_size::Int, n_steps::Int64, learning_rate::Float64, decay_rate::Float64, index::Int)
        mypath = joinpath(path, "learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n-steps=$(n_steps)_tuple-size=$(tuple_size)")
        mkpath(mypath)
        filename = joinpath(mypath, "checkpoint_$(index).csv")

        env = get_enviroment(df, vmodel, models, encoders, n_steps, learning_rate, decay_rate)
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
                    G = G + learning_rate^n_steps * WQNN.predict(env.V, BinInput(env, S[t_]))
                end
                alpha = G - WQNN.predict(env.V, BinInput(env, S[t_]))

                WQNN.train!(
                    env.V,
                    BinInput(env, S[t_]), 
                    G
                ) 

                grad = env.learning_rate * G * alpha * grad_ln_prob(env, S[t_], A[t_])
                for a in 1:length(env.fmodels)
                    g = grad[a]
                    if isnan(g)
                        g = 0.
                    end
                    GDWPNN.train!(env.fmodels[a], BinInput(env, S[t_]), g)
                end
            end
            t = t + 1      
        end

        if (index%25 == 0)|(index>(2500-25))
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
        V::WQNN.Model
        fmodels::Array{GDWPNN.Model}
        encoders::Dict{String, thermometer.Thermometer}
        state::State
        n_steps::Float64
        learning_rate::Float64
        decay_rate::Float64
    end


    function get_enviroment(df, Vmodel::WQNN.Model, Fmodels::Array{GDWPNN.Model}, encoders::Dict{String, thermometer.Thermometer}, n_steps::Int64, learning_rate::Float64, decay_rate::Float64)
        nans = [NaN for i in 1:nrow(df)]
        df.angle = nans
        df.POA = nans
        df.reward = nans
            
        return Enviroment(df, 
            Vmodel,
            Fmodels,
            encoders,
            State(1,0),
            n_steps,
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

        return (irrad + bonus)^3/(1200^3), irrad
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

    function grad_ln_prob(env::Enviroment, state::State, action::Int64)
        probs = PolicyProbs(env, state)
        # k = k_distance(env, state)
        n_actions = length(env.fmodels)

        return [a != action ? -probs[a] : (1-probs[a]) for a in 1:n_actions]
    end

    function k_distance(env::Enviroment, state::State)
        n_actions = length(env.fmodels)
        return [GDWPNN.get_k(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
    end


    function PolicyProbs(env::Enviroment, state::State)
        n_actions = length(env.fmodels)
        fs = [GDWPNN.get_f(env.fmodels[action], BinInput(env, state)) for action in 1:n_actions]
        exp_f = [exp(fs[action]) for action in 1:n_actions]
        s = sum(exp_f)
            
        if s > 1e-4
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
end