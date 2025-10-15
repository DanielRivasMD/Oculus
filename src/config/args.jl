####################################################################################################

using ArgParse
using TOML

####################################################################################################

function struct_to_dict(x)
    Dict(name => getfield(x, name) for name in propertynames(x))
end

function load_cnnparams(path::String)
    params = CNNParams()
    cnn_cfg = path != "" ? TOML.parsefile(path) : Dict()

    if haskey(cnn_cfg, "device")
        cnn_cfg["device"] = cnn_cfg["device"] == "gpu" ? gpu : cpu
    end
    if haskey(cnn_cfg, "σ")
        cnn_cfg["σ"] = cnn_cfg["σ"] == "relu" ? relu : tanh
    end

    return CNNParams(; merge(struct_to_dict(params), cnn_cfg)...)
end

function load_sampleparams(path::String)
    params = SampleParams()
    sample_cfg = path != "" ? TOML.parsefile(path) : Dict()
    return SampleParams(; merge(struct_to_dict(params), sample_cfg)...)
end

function cnn_args()
    desc =
        "\e[1;32mDaniel Rivas\e[0m " *
        "\e[3;90m<danielrivasmd@gmail.com>\e[0m\n\n\n\n\n" *
        "Train Oculus, CNN for ancient DNA identification\n"

    s = ArgParseSettings(description = desc)

    @add_arg_table s begin
        "--cnn"
            help = "Path to CNNParams TOML"
            default = ""

        "--sample"
            help = "Path to SampleParams TOML"
            default = ""
    end

    return parse_args(s)
end

####################################################################################################
