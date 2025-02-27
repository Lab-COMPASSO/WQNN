using DataFrames, Random, CSV
include("utils/SARSA.jl")
using .forget_model

evaluation = [
    1, 16, 31, 46, 61, 76, 91, 106, 
    121, 136, 151, 166, 181, 196, 211,
    226, 241, 256, 271, 286, 301, 316,
    331, 346, 361
    ]


for epsilon in [0.25, 0.5, 0.7, 0.9,0.98]
    for decay_rate  in [0.25, 0.5, 0.7, 0.9,0.98]
        for learning_rate in [0.98, 0.9, 0.7, 0.5, 0.25]
            for n_steps in [1, 2, 4, 8, 16]
             for tuple_size in [10, 20, 40, 80, 160, 320, 720]
                    day = Int(24*(60/15))
                    df = DataFrame(CSV.File("C:/Users/guien/OneDrive/Mestrado/Tese/src/data/0_case=0.csv"))

                    # try
                        if ~isfile("./results/CS/SARSA/epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n_steps=$(n_steps)_tuple-size=$(tuple_size)/checkpoint_2500.csv")
                
                            models = [forget_model.generate_Model(1320, tuple_size, learning_rate) for i in 1:5]
                
                            encoders = get_encoders(df)
                            @info ("epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_n_steps=$(n_steps)_tuple-size=$(tuple_size)")
                        
                            for index in 1:5000
                                @time begin
                                    j = Int(rand(1:364))
                                    i = (j-1)*day + 1
                                
                                    run_episode(df[i:(i + day),:], "./results/CS/SARSA/", models, encoders, tuple_size, n_steps, epsilon, learning_rate, decay_rate, index)
                                end
                            end
                            counter = 1
                            for j in evaluation
                                @time begin
                                    i = (j-1)*day + 1
                                    run_episode(df[i:(i + day),:], "./results/CS/SARSA/evaluation/", models, encoders, tuple_size, n_steps, epsilon, learning_rate, decay_rate, counter)
                                    counter = counter + 1
                                end
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
