####################################################################################################

using ArgParse
using TOML
using FilePathsBase: basename, splitext, joinpath, dirname, isabspath, isdir, mkpath

####################################################################################################

const HELP =
  "\e[1;32mDaniel Rivas\e[0m " * "\e[3;90m<danielrivasmd@gmail.com>\e[0m\n\n\n\n\n"

####################################################################################################

"Convert struct fields to a Dict with Symbol keys"
function struct_to_dict(x)
  Dict(name => getfield(x, name) for name in propertynames(x))
end

"Convert Dict with String keys (from TOML) into Dict with Symbol keys"
function symbolise_keys(d::Dict)
  Dict(Symbol(k) => v for (k, v) in d)
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
  desc = HELP * "Train Oculus, CNN for ancient DNA identification\n"

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
  desc = HELP * "Performance reporting for Oculus CNN training runs\n"

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
  desc = HELP * "Inference with Oculus CNN on ancient DNA samples\n"

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
    data_root = splitext(basename(args["data"]))[1]
    model_root = splitext(basename(args["model"]))[1]
    args["out"] = "$(data_root)_$(model_root).csv"
  end

  return args
end

####################################################################################################

####################################################################################################
# Args API
####################################################################################################

"""
    roc_args() -> Dict

Parse CLI arguments for ROC curve plotting.

Options:
--modern   : CSV file with modern predictions
--ancient  : CSV file with ancient predictions
--single   : Single CSV file with ground truth in 4th column
--out      : Output HTML file (default: graph/roc/roc.html)
"""
function roc_args()
  s = ArgParseSettings()

  @add_arg_table s begin
    "--modern"
    help = "CSV file with modern predictions"
    arg_type = String
    required = false   # set true if you want to force two-file mode
    "--ancient"
    help = "CSV file with ancient predictions"
    arg_type = String
    required = false
    "--single"
    help = "Single CSV file with ground truth in 4th column"
    arg_type = String
    required = false
    "--out"
    help = "Output HTML file (default: graph/roc/roc.html)"
    arg_type = String
    default = joinpath(Paths.ROC, "roc.html")
  end

  return parse_args(s)
end

####################################################################################################

function sysimage_args()
  desc = HELP * "Build a custom Julia sysimage for a script\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "script"
    help = "Path to the Julia script to analyze"
    arg_type = String

    "--out"
    help = "Output sysimage path (default = <scriptname>.so)"
    default = nothing

    "--middleman"
    help = "Use an intermediary precompile driver file (default = false)"
    action = :store_true

    "--exclude"
    help = "Comma-separated list of packages to exclude"
    arg_type = String
    default = ""
  end

  args = parse_args(s)

  # Normalize exclude into a Vector{String}
  excl = isempty(args["exclude"]) ? String[] : split(args["exclude"], ',') .|> strip
  args["exclude"] = excl

  return args
end

####################################################################################################
