####################################################################################################

using ArgParse
using TOML
using FilePathsBase: basename, splitext, joinpath, dirname, isabspath, isdir, mkpath

####################################################################################################

"Convert struct fields to a Dict with Symbol keys"
function struct_to_dict(x)
    Dict(name => getfield(x, name) for name in propertynames(x))
end

"Convert Dict with String keys (from TOML) into Dict with Symbol keys"
function symbolise_keys(d::Dict)
    Dict(Symbol(k) => v for (k,v) in d)
end

function load_cnnparams(path::String)
    params = CNNParams()
    cnn_cfg = path != "" ? symbolise_keys(TOML.parsefile(path)["cnn"]) : Dict()

    if haskey(cnn_cfg, :device)
        cnn_cfg[:device] = cnn_cfg[:device] == "gpu" ? gpu : cpu
    end
    if haskey(cnn_cfg, :σ)
        cnn_cfg[:σ] = cnn_cfg[:σ] == "relu" ? relu : tanh
    end

    return CNNParams(; merge(struct_to_dict(params), cnn_cfg)...)
end

function load_sampleparams(path::String)
    params = SampleParams()
    sample_cfg = path != "" ? symbolise_keys(TOML.parsefile(path)["sample"]) : Dict()
    return SampleParams(; merge(struct_to_dict(params), sample_cfg)...)
end

####################################################################################################

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

function perf_args()
    desc =
        "\e[1;32mDaniel Rivas\e[0m " *
        "\e[3;90m<danielrivasmd@gmail.com>\e[0m\n\n\n\n\n" *
        "Performance reporting for Oculus CNN training runs\n"

    s = ArgParseSettings(description = desc)

    @add_arg_table s begin
        "--model"
            help = "Path to trained CNN BSON checkpoint"
            arg_type = String
            required = true

        "--out"
            help = "Root name for performance report (HTML). If not provided, defaults to graph/performance/<model>.html"
            arg_type = String
            default = ""
    end

    args = parse_args(s)

    # Ensure graph/performance directory exists
    outdir = joinpath("graph", "performance")
    if !isdir(outdir)
        mkpath(outdir)
    end

    if args["out"] == ""
        # Default: model basename with .html inside graph/performance
        model_root = splitext(basename(args["model"]))[1]
        args["out"] = joinpath(outdir, "$(model_root).html")
    else
        # If user gave a bare name, place it under graph/performance
        root, ext = splitext(args["out"])
        fname = ext == ".html" ? args["out"] : "$(root).html"
        if isabspath(fname) || startswith(fname, "graph/")
            args["out"] = fname
        else
            args["out"] = joinpath(outdir, fname)
        end
    end

    return args
end

####################################################################################################

function infer_args()
    desc =
        "\e[1;32mDaniel Rivas\e[0m " *
        "\e[3;90m<danielrivasmd@gmail.com>\e[0m\n\n\n\n\n" *
        "Inference with Oculus CNN on ancient DNA samples\n"

    s = ArgParseSettings(description = desc)

    @add_arg_table s begin
        "--model"
            help = "Path to trained CNN BSON checkpoint"
            arg_type = String
            required = true

        "--data"
            help = "Path to input data file for inference"
            arg_type = String
            required = true

        "--out"
            help = "Path to save predictions (CSV). If not provided, will be auto‑generated."
            arg_type = String
            default = ""
    end

    args = parse_args(s)

    # If no --out given, build default name: <data_root>_<model_root>.csv
    if args["out"] == ""
        data_root  = splitext(basename(args["data"]))[1]
        model_root = splitext(basename(args["model"]))[1]
        args["out"] = "$(data_root)_$(model_root).csv"
    end

    return args
end

####################################################################################################
