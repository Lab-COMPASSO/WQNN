


 using DataFrames, Random
 include("utils/REINFORCE.jl")
 using .policy_model

 evaluation = [
    1, 16, 31, 46, 61, 76, 91, 106, 
    121, 136, 151, 166, 181, 196, 211,
    226, 241, 256, 271, 286, 301, 316,
    331, 346, 361
    ]


for tuple_size in [160, 320, 720, 10, 20, 40, 80]
    for decay_rate  in [0.5, 0.7, 0.9,0.98, 0.25 ]
        for learning_rate in [0.9, 0.7, 0.5, 0.25, 0.98]
                day = Int(24*(60/15))
                df = DataFrame(CSV.File("C:/Users/guien/OneDrive/Mestrado/Tese/src/data/real_scenario.csv"))
                # try
                    if ~isfile("./results/v02/REINFORCE/learning-rate=$(learning_rate)_decay-rate=$(decay_rate)tuple-size=$(tuple_size)/checkpoint_5000.csv")
                        basemodel = policy_model.generate_Model(1320, tuple_size)
                        models = [deepcopy(basemodel) for i in 1:5]

                        encoders = get_encoders(df)
                        println("learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)")
                    
                        for index in 1:5000
                            @time begin
                                j = Int(rand(1:364))
                                i = (j-1)*day + 1
                            
                                run_episode(df[i:(i + day),:], "./results/v02/REINFORCE/", models, encoders, tuple_size, learning_rate, decay_rate, index)
                            end
                        end
                        counter = 1
                        for j in evaluation
                            @time begin
                                i = (j-1)*day + 1
                                run_episode(df[i:(i + day),:], "./results/v02/REINFORCE/evaluation/", models, encoders, tuple_size, learning_rate, decay_rate, counter)
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
