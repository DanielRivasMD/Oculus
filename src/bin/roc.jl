####################################################################################################
# Imports
####################################################################################################

using DelimitedFiles
using ArgParse
using Statistics
using Plots
using Weave
using FilePathsBase: joinpath

####################################################################################################
# Load configuration
####################################################################################################

begin
  # Load path definitions
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.CONFIG, "args.jl"))   # Args API (now includes roc_args)
end

####################################################################################################
# Helper functions
####################################################################################################

function load_probs_labels(
  path::String;
  has_truth::Bool = false,
  label::Union{Nothing,Int} = nothing,
)
  data, header = readdlm(path, ',', header = true)
  probs = Float64.(data[:, 3])  # p1 column
  if has_truth
    labels = Int.(data[:, 4])  # use provided ground truth
  elseif label !== nothing
    labels = fill(label, length(probs))  # assign explicit label
  else
    labels = nothing
  end
  return probs, labels
end

function roc_curve(probs::Vector{Float64}, labels::Vector{Int}, nbins::Int = 100)
  thresholds = range(0, 1, length = nbins)
  tpr = Float64[]
  fpr = Float64[]
  P = sum(labels .== 1)
  N = sum(labels .== 0)
  for τ in thresholds
    preds = probs .>= τ
    TP = sum((preds .== 1) .& (labels .== 1))
    FP = sum((preds .== 1) .& (labels .== 0))
    push!(tpr, TP / max(P, 1))
    push!(fpr, FP / max(N, 1))
  end
  return collect(fpr), collect(tpr)
end

####################################################################################################
# CLI
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing
  args = roc_args()

  if args["single"] !== nothing
    probs, labels = load_probs_labels(args["single"]; has_truth = true)
    fpr, tpr = roc_curve(probs, labels)
    plt = plot(
      fpr,
      tpr,
      xlabel = "False Positive Rate",
      ylabel = "True Positive Rate",
      title = "ROC (single file)",
      legend = false,
    )

  elseif args["modern"] !== nothing && args["ancient"] !== nothing
    modern_probs, modern_labels = load_probs_labels(args["modern"]; label = 0)
    ancient_probs, ancient_labels = load_probs_labels(args["ancient"]; label = 1)
    probs = vcat(modern_probs, ancient_probs)
    labels = vcat(modern_labels, ancient_labels)
    fpr, tpr = roc_curve(probs, labels)
    plt = plot(
      fpr,
      tpr,
      xlabel = "False Positive Rate",
      ylabel = "True Positive Rate",
      title = "ROC (modern vs ancient)",
      legend = false,
    )

  else
    error("Must provide either --single or both --modern and --ancient")
  end

  outpath = args["out"]
  mdfile = joinpath(Paths.ROC, "roc_report.jmd")

open(mdfile, "w") do io
    println(io, """
# ROC Curve Report

```julia
using Plots
plot($fpr, $tpr,
     xlabel="False Positive Rate",
     ylabel="True Positive Rate",
     title="ROC Curve",
     legend=false)
```
""",
    )
  end

  weave(mdfile, out_path = outpath, doctype = "md2html")
  rm(mdfile; force=true) # remove the intermediary file
  println("ROC curve report written to $outpath")
end

####################################################################################################
