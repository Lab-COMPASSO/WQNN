using DataFrames, Random, CSV
include("utils/DEEPQ.jl")
using .deepQ_model

evaluation = [
    1, 16, 31, 46, 61, 76, 91, 106, 
    121, 136, 151, 166, 181, 196, 211,
    226, 241, 256, 271, 286, 301, 316,
    331, 346, 361
    ]


for epsilon in Vector{Float32}([0.25, 0.5, 0.7, 0.9,0.98])
    for decay_rate  in Vector{Float32}([0.25, 0.5, 0.7, 0.9,0.98])
        for learning_rate in Vector{Float32}([0.98, 0.9, 0.7, 0.5, 0.25])
            for hidden_layer_size in Vector{Int64}([8, 16, 32, 64, 128])
                    day = Int(24*(60/15))
                    df = DataFrame(CSV.File("C:/Users/guien/OneDrive/Mestrado/Tese/src/data/real_scenario.csv"))

                    # try
                        if ~isfile("./results/v02/DEEPQ/epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_hidden-size=$(hidden_layer_size)/checkpoint_2500.csv")
                
                            model = deepQ_model.generate_Model(4,5, hidden_layer_size)
                            @info ("epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_hidden-size=$(hidden_layer_size)")
                        
                            for index in 1:5000
                                @time begin
                                    j = Int(rand(1:364))
                                    i = (j-1)*day + 1
                                
                                    run_episode(df[i:(i + day),:], "./results/v02/DEEPQ/", model, hidden_layer_size, epsilon, learning_rate, decay_rate, index)
                                end
                            end
                            counter = 1
                            for j in evaluation
                                @time begin
                                    i = (j-1)*day + 1
                                    run_episode(df[i:(i + day),:], "./results/v02/DEEPQ/evaluation/", model, hidden_layer_size, epsilon, learning_rate, decay_rate, index)
                                    counter = counter + 1
                                end
                            end
                        
                    # catch y
                    #     @info ("Exception: ", y) # What to do on error.
                    # end
                end
            end
        end
    end
end
