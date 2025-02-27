module thermometer
    abstract type Thermometer end

    using Statistics

    mutable struct DistributiveEncoder <: Thermometer
        bins
        quantiles
    end

    mutable struct LinearEncoder <: Thermometer
        min
        max
        bins
    end

    mutable struct CircularEncoder <: Thermometer
        min
        max
        
        k
        
        marker_size
        bins
    end

    function fit!(encoder::LinearEncoder, vals)::LinearEncoder
        encoder.min = minimum(vals)
        encoder.max = maximum(vals)
        
        return encoder
    end

    function encode(encoder::LinearEncoder, val::Float64)::BitArray
        k = (encoder.max - encoder.min)/encoder.bins
        
        a = val - encoder.min
        # println(k, " - ", a)
        return BitArray([i <= (ceil(a/(k))) for i in 1:encoder.bins])
    end

    function fit!(encoder::DistributiveEncoder, vals::Vector)::DistributiveEncoder
        k = 1/encoder.bins
        encoder.quantiles = [quantile(vals, k*(i)) for i in 1:encoder.bins]
            
        return encoder
    end

    function encode(encoder::DistributiveEncoder, val::Float64)::BitArray

        return BitArray([val >= encoder.quantiles[i] for i in 1:encoder.bins])
    end

    function fit!(encoder::CircularEncoder, vals)::CircularEncoder
        encoder.min = minimum(vals)
        encoder.max = maximum(vals)
        
        encoder.k = (encoder.max - encoder.min)/encoder.bins

        return encoder
    end

    function encode(encoder::CircularEncoder, var::Vector)::BitArray
        delta = var - encoder.min
        i = Int(ceil(delta/encoder.k))
        while i > encoder.bins
            i -= encoder.bins
        end

        while i < 1
            i += encoder.bins
        end

        j = Int(i + encoder.marker_size)

        while j > encoder.bins
            j -= encoder.bins
        end

        while j < 1
            j += encoder.bins
        end

        # println(i, j)
        if j > i
            return BitArray([(n >= i && n < j) for n in 1:encoder.bins])

        else
            return BitArray([(n >= i || n < j) for n in 1:encoder.bins])
        end
    end
end


module policy_model
    using Random
    Random.seed!(971)

    mutable struct RAM_memory
        c::Int64
        sum::Float64
    end

    mutable struct RAM
        max_c::Int64
        tuple_size::Int
        table::Dict{String, RAM_memory}
    end

    function get_position(input_array::BitArray)::String
        return join(string.(input_array))
    end

    function generate_RAM(tuple_size::Int)::RAM
        return RAM(0, tuple_size, Dict{Int, RAM_memory}())
    end
        
    function put!(ram::RAM, input_array::BitArray, y::Float64)::RAM_memory
        position = get_position(input_array)

        x = get_memory(ram, input_array)
        x.sum = x.sum + y
        x.c = x.c + 1

        if x.c > ram.max_c
            ram.max_c = Int64(x.c)
        end

        ram.table[position] = x
        
    end    
        
    function get_memory(ram::RAM, input_array::BitArray)::RAM_memory
        position = get_position(BitArray(input_array))
        return get(ram.table, position, RAM_memory(0, 0))
    end

    struct AddressMapping
        input_size::Int 
        tuple_size::Int
        n_rams::Int

        mapping::Array{Int}

    end

    function generate_Addressing(input_size::Int, tuple_size::Int)::AddressMapping
        return AddressMapping(input_size, tuple_size, ceil(input_size/tuple_size), shuffle(collect(1:input_size)))
    end

    function get_addresses(addressing::AddressMapping, input_array::BitArray)::Array{BitArray}
        output = []
        for n in collect(1:addressing.n_rams)
            i = (n-1)*addressing.tuple_size+1
            j = minimum([(n)*addressing.tuple_size, addressing.input_size])
            x = addressing.mapping[i:j]
            y = input_array[x]
            push!(output,y)
        end
        
        return output
    end


    struct Model
        input_size::Int 
        tuple_size::Int
        
        n_rams::Int
        RAMs::Array{RAM}
        
        AddMappping::AddressMapping
    end

    function generate_Model(input_size::Int, tuple_size::Int)::Model
        n_rams = Int(ceil(input_size/tuple_size))
        
        if n_rams == floor(input_size/tuple_size)
            RAMs = [generate_RAM(tuple_size) for i in 1:n_rams]
        else
            RAMs = [generate_RAM(tuple_size) for i in 1:(n_rams - 1)]
            push!(RAMs, generate_RAM(input_size % tuple_size))
        end
    return Model(
                input_size, tuple_size,
                n_rams, RAMs, generate_Addressing(input_size, tuple_size)
                )       
    end

    function train!(model::Model, bin_input::BitArray, y::Float64)
        mapped_input = get_addresses(model.AddMappping, bin_input)
        for i in 1:model.n_rams
            # println(i," - ",model.RAMs[i]," - ",mapped_input[i])
            put!(model.RAMs[i],mapped_input[i], y)
        end
    end
        
    function get_f(model::Model, bin_input::BitArray)::Float64
        c, s = 0, 0
        mapped_input = get_addresses(model.AddMappping, bin_input)

        for i in 1:model.n_rams
            x = get_memory(model.RAMs[i], mapped_input[i])
            s+= x.sum
        end
        if isnan(s)
            s = 0        
        end
        return s/model.n_rams
    end 

    function get_k(model::Model, bin_input::BitArray)::Float64
        c, s = 0, 0
        mapped_input = get_addresses(model.AddMappping, bin_input)

        for i in 1:model.n_rams
            x = get_memory(model.RAMs[i], mapped_input[i])
            tmp = model.RAMs[i].max_c > 0 ? x.c/model.RAMs[i].max_c : 0
            c = c + tmp
        end
        
        return c/model.n_rams
    end 

    
end

module forget_model
    using Random
    Random.seed!(971)

    mutable struct RAM_memory
        c::Int
        sum::Float64
    end

    struct RAM
        tuple_size::Int
        forget_factor::Float64
        table::Dict{String, RAM_memory}
    end

    function get_position(input_array::BitArray)::String
        return join(string.(input_array))
    end

    function generate_RAM(tuple_size::Int, forget_factor::Float64)::RAM
        return RAM(tuple_size, forget_factor, Dict{Int, RAM_memory}())
    end
        
    function put!(ram::RAM, input_array::BitArray, y::Float64)::RAM_memory
        position = get_position(input_array)

        x = get_memory(ram, input_array)
        x.c += 1
        x.sum = ram.forget_factor * x.sum + y

        ram.table[position] = x
    end    
        
    function get_memory(ram::RAM, input_array::BitArray)::RAM_memory
        position = get_position(BitArray(input_array))
        return get(ram.table, position, RAM_memory(0, 0))
    end

    struct AddressMapping
        input_size::Int 
        tuple_size::Int
        n_rams::Int

        mapping::Array{Int}

    end

    function generate_Addressing(input_size::Int, tuple_size::Int)::AddressMapping
        return AddressMapping(input_size, tuple_size, ceil(input_size/tuple_size), shuffle(collect(1:input_size)))
    end

    function get_addresses(addressing::AddressMapping, input_array::BitArray)::Array{BitArray}
        output = []
        for n in collect(1:addressing.n_rams)
            i = (n-1)*addressing.tuple_size+1
            j = minimum([(n)*addressing.tuple_size, addressing.input_size])
            x = addressing.mapping[i:j]
            y = input_array[x]
            push!(output,y)
        end
        
        return output
    end


    struct Model
        input_size::Int 
        tuple_size::Int
        forget_factor::Float64
        
        n_rams::Int
        RAMs::Array{RAM}
        
        AddMappping::AddressMapping
    end

    function generate_Model(input_size::Int, tuple_size::Int, forget_factor::Float64)::Model
        n_rams = Int(ceil(input_size/tuple_size))
        
        if n_rams == floor(input_size/tuple_size)
            RAMs = [generate_RAM(tuple_size, forget_factor) for i in 1:n_rams]
        else
            RAMs = [generate_RAM(tuple_size, forget_factor) for i in 1:(n_rams - 1)]
            push!(RAMs, generate_RAM(input_size % tuple_size, forget_factor))
        end
    return Model(
                input_size, tuple_size, forget_factor,
                n_rams, RAMs, generate_Addressing(input_size, tuple_size)
                )       
    end

    function train!(model::Model, bin_input::BitArray, y::Float64)
        mapped_input = get_addresses(model.AddMappping, bin_input)
        for i in 1:model.n_rams
            # println(i," - ",model.RAMs[i]," - ",mapped_input[i])
            put!(model.RAMs[i],mapped_input[i], y)
        end
    end
        
    function predict(model::Model, bin_input::BitArray)::Float64
        c, s = 0, 0
        mapped_input = get_addresses(model.AddMappping, bin_input)

        for i in 1:model.n_rams
            x = get_memory(model.RAMs[i], mapped_input[i])
            c+= (1 - model.forget_factor^x.c)/(1 - model.forget_factor)
            s+= x.sum
        end
        
        if c > 0
            return s/c
        else
            return 0
        end
    end 
end

module deepQ_model
    using Flux
    mutable struct Model
        model::Chain
        optim::Flux.Optimise.Adam
    end

    function generate_Model(input_size::Int64, n_actions::Int64, hidden_layer_size::Int64) ::Model
        model = Chain(
            Dense(input_size => hidden_layer_size, relu; init=Flux.glorot_uniform(gain=10)),   # activation function inside layer
            Dense(hidden_layer_size => n_actions, relu; init=Flux.glorot_uniform(gain=10)),
            relu)

        optim = Flux.Adam(0.1)

        return Model(model, optim)
        
    end

    function train!(model::Model, x::Vector{Float32}, y::Vector{Float32})
        opt_state = Flux.setup( model.optim, model.model)
        Flux.train!(model.model, [(x, y)], opt_state) do m, x, y
            sum((m(x)- y).^2)
        end
    end

    function predict(model::Model, input::Vector{Float32})::Vector{Float32}
        return model.model(input)
    end
end