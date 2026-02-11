####################################################################################################

using ArgParse

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

####################################################################################################

function extract_args()
  desc = HELP * "Extract Illumina‑like reads from a genome FASTA\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "--genome"
    help = "Input genome FASTA file"
    arg_type = String
    required = true

    "--out"
    help = "Output FASTA file for simulated reads"
    arg_type = String
    default = "simulated_reads.fasta"

    "--num_reads"
    help = "Number of reads to simulate (default: 1000)"
    arg_type = Int
    default = 1000

    "--read_len"
    help = "Length of each simulated read (default: 76)"
    arg_type = Int
    default = 76
  end

  return parse_args(s)
end

####################################################################################################

function deamination_args()
  desc =
    HELP *
    "Compare per‑position nucleotide composition between modern and ancient FASTA files\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "--modern"
    help = "Modern FASTA file"
    arg_type = String
    required = true

    "--ancient"
    help = "Ancient FASTA file"
    arg_type = String
    required = true

    "--csv"
    help = "Output CSV file (default: out.csv)"
    arg_type = String
    default = "out.csv"

    "--png"
    help = "Output PNG plot file (default: out.png)"
    arg_type = String
    default = "out.png"

    "--verbose"
    help = "Print detailed information"
    action = :store_true
  end

  args = parse_args(s)
  return args
end


####################################################################################################

function feature_args()
  desc = HELP * "Feature engineering for ancient vs modern FASTA sequences\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "--modern"
    help = "Modern FASTA file"
    arg_type = String
    required = true

    "--ancient"
    help = "Ancient FASTA file"
    arg_type = String
    required = true

    "--out"
    help = "Output CSV file for engineered features"
    arg_type = String
    default = "features.csv"

    "--heavy"
    help = "Use heavy one-hot encoding for every position"
    action = :store_true
  end

  return parse_args(s)
end

####################################################################################################

function regression_args()
  desc = HELP * "Run a linear regression model on engineered features\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "--in"
    help = "Input CSV file containing engineered features"
    arg_type = String
    required = true

    "--out"
    help = "Optional output CSV for predictions"
    arg_type = String
    default = nothing
  end

  return parse_args(s)
end

####################################################################################################

function cnn_args()
  desc = HELP * "Train Oculus, CNN for ancient DNA identification\n"
  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "--cnn"
    help = "Path to CNNParams TOML"

    "--sample"
    help = "Path to SampleParams TOML"
  end

  return parse_args(s)
end

####################################################################################################

function performance_args()
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

function inference_args()
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
