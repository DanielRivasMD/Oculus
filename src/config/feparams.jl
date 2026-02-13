####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    FeatureEngineeringParams(; kwargs...)

Hyperparameter container for FASTA → feature engineering.

# Fields
- `modern::String`        Path to modern FASTA file
- `ancient::String`       Path to ancient FASTA file
- `out::String = "features.csv"`
- `onehot::Bool = false`  Whether to use heavy one‑hot encoding (per‑position)
"""
@with_kw mutable struct FeatureEngineeringParams
  modern::String = ""
  ancient::String = ""
  out::String = "features.csv"
  onehot::Bool = false
end

####################################################################################################

function loadFeatureParams(path::Union{Nothing,String}, args)
  params = loadParams(path, FeatureEngineeringParams; section = :features)

  overrides = Dict(
    :modern => args["modern"],
    :ancient => args["ancient"],
    :out => args["out"],
    :onehot => args["onehot"],
  )

  merged = merge(struct_to_dict(params), overrides)
  return FeatureEngineeringParams(; merged...)
end

####################################################################################################
