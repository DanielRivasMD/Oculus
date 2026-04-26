module RCCore

####################################################################################################
using DelimitedFiles
using Plots
using Weave
using FilePathsBase: joinpath

export load_probs_labels, roc_curve, generate_roc_report

####################################################################################################

"""
    load_probs_labels(path::String; has_truth=false, label=nothing)

Read a CSV with columns `id,p0,p1` (and optionally a 4th column for ground truth).
Returns `(probs::Vector{Float64}, labels::Union{Vector{Int},Nothing})`.
If `has_truth=false`, a fixed `label` must be supplied.
"""
function load_probs_labels(
  path::String;
  has_truth::Bool = false,
  label::Union{Nothing,Int} = nothing,
)
  data, header = readdlm(path, ',', header = true)
  probs = Float64.(data[:, 3])   # p1 column

  if has_truth
    labels = Int.(data[:, 4])
  elseif label !== nothing
    labels = fill(label, length(probs))
  else
    labels = nothing
  end
  return probs, labels
end

"""
    roc_curve(probs::Vector{Float64}, labels::Vector{Int}; nbins=100) -> (fpr, tpr)

Compute ROC curve points (false positive rate, true positive rate).
"""
function roc_curve(probs::Vector{Float64}, labels::Vector{Int}; nbins::Int = 100)
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

end
