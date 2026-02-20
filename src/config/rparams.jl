####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    RegressionParams(; kwargs...)

Hyperparameter container for logistic, ridge, lasso, and elastic‑net regression.

# Fields
- `infile::String`     Input CSV file
- `out::Union{String,Nothing} = nothing`
- `reg::String = "none"`  
    "none" | "ridge" | "lasso" | "elasticnet"

- `alpha::Float64 = 0.5`  
    Elastic‑net mixing parameter (0 < α < 1)

- `nfolds::Int = 10`  
    Number of CV folds for GLMNet

- `split::Float64 = 0.0`  
    Fraction of data to use as test set (0.0 = no split)
"""
@with_kw mutable struct RegressionParams
  infile::String
  out::Union{String,Nothing} = nothing
  reg::String = "none"
  alpha::Float64 = 0.5
  nfolds::Int = 10
  split::Float64 = 0.0
end

####################################################################################################

function loadRegressionParams(path::String)
  params = RegressionParams(infile = "")
  cfg = path != "" ? symbolise_keys(TOML.parsefile(path)["regression"]) : Dict()
  return RegressionParams(; merge(struct_to_dict(params), cfg)...)
end

####################################################################################################
