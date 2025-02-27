


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
    for encoder_size in [128, 256, 512, 1024, 2048, 4096]
    
        decay_rate = 0.7
        learning_rate = 0.25
            day = Int(24*(60/15))
            df = DataFrame(CSV.File("C:/Users/guien/OneDrive/Mestrado/Tese/src/data/real_scenario.csv"))
            # try
                if ~isfile("./results/encoder/REINFORCE/learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)_encoder-size=$(encoder_size)/checkpoint_5000.csv")
                    basemodel = policy_model.generate_Model(encoder_size, tuple_size)
                    models = [deepcopy(basemodel) for i in 1:5]

                    encoders = get_encoders(df, encoder_size)
                    println("learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_tuple-size=$(tuple_size)_encoder-size=$(encoder_size)")
                
                    for index in 1:5000
                        @time begin
                            j = Int(rand(1:364))
                            i = (j-1)*day + 1
                        
                            run_episode(df[i:(i + day),:], "./results/encoder/REINFORCE/", models, encoders, tuple_size, learning_rate, decay_rate, encoder_size, index)
                        end
                    end
                    counter = 1
                    for j in evaluation
                        @time begin
                            i = (j-1)*day + 1
                            run_episode(df[i:(i + day),:], "./results/encoder/REINFORCE/evaluation/", models, encoders, tuple_size, learning_rate, decay_rate, encoder_size, index)
                            counter = counter + 1
                        end
                    end
                end
            # catch y
            #     @info ("Exception: ", y) # What to do on error.
            # end
        end
    end
    
