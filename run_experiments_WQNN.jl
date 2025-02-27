
using DataFrames, Random
include("utils/WQNN.jl")
include("utils/WQNN_enviroment.jl")

df = DataFrame(CSV.File("C:/Users/guien/OneDrive/Mestrado/Tese/src/data/real_scenario.csv"))

day = Int(24*(60/15))


evaluation = [
    1, 16, 31, 46, 61, 76, 91, 106, 
    121, 136, 151, 166, 181, 196, 211,
    226, 241, 256, 271, 286, 301, 316,
    331, 346, 361
    ]

PATH = "./results/v02/WQNN"

params = []

for epsilon in [0.25, 0.5, 0.7, 0.9,0.98]
    for decay_rate  in [0.25, 0.5, 0.7, 0.9,0.98]
        for learning_rate in [0.98, 0.9, 0.7, 0.5, 0.25]
            for forget_factor in [0.25, 0.5, 0.7, 0.9,0.98]
                for tuple_size in [10, 20, 40, 80, 160, 320, 720]

                    push!(params, (epsilon, decay_rate, learning_rate, forget_factor, tuple_size))
                end
            end
        end
    end
end

for p in shuffle(params)
    epsilon, decay_rate, learning_rate, forget_factor, tuple_size  = p

    try
        if ~isfile("$(PATH)/epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)/checkpoint_2500.csv")

            models = [generate_Model(1320, tuple_size, forget_factor) for i in 1:5]

            encoders = get_encoders(df)
            println("epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)")
        
            for index in 1:5000
                @time begin
                    j = Int(rand(1:364))
                    i = (j-1)*day + 1
                
                    run_episode(df[i:(i + day),:], PATH, models, encoders, tuple_size, forget_factor, epsilon, learning_rate, decay_rate, index)
                end
            end
            counter = 1
            for j in evaluation
                @time begin
                    i = (j-1)*day + 1
                    run_episode_eval(df[i:(i + day),:], "$(PATH)/evaluation/", models, encoders, tuple_size, forget_factor, epsilon, learning_rate, decay_rate, counter)
                    counter = counter + 1
                end
            end
        end
    catch y
        @info ("Exception: ", y) # What to do on error.
    end
    
end
