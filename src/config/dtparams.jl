####################################################################################################

using Parameters: @with_kw

####################################################################################################

"""
    DecisionTreeParams(; kwargs...)

Hyperparameter container for Decision Tree, Random Forest, and XGBoost classifiers.

# Fields
- `model::String = "tree"`  
    Which model to run: `"tree" | "forest" | "xgboost"`

- `split::Float64 = 0.2`  
    Fraction of data to use as test set (0.0 = no split)

- `seed::Int = 42`  
    Random seed for reproducibility

### Decision Tree / Random Forest
- `max_depth::Int = 6`
- `min_samples_leaf::Int = 5`

### Random Forest
- `n_trees::Int = 100`
- `rf_partial_sampling::Float64 = 0.7`

### XGBoost
- `xgb_rounds::Int = 200`
- `xgb_eta::Float64 = 0.1`
- `xgb_max_depth::Int = 6`
- `xgb_subsample::Float64 = 0.8`
- `xgb_colsample_bytree::Float64 = 0.8`
"""
@with_kw mutable struct DecisionTreeParams
  model::String = "tree"

  split::Float64 = 0.2
  seed::Int = 1

  # Tree / Forest shared
  max_depth::Int = 6
  min_samples_leaf::Int = 5

  # Random Forest
  n_trees::Int = 100
  rf_partial_sampling::Float64 = 0.7

  # XGBoost
  xgb_rounds::Int = 200
  xgb_eta::Float64 = 0.1
  xgb_max_depth::Int = 6
  xgb_subsample::Float64 = 0.8
  xgb_colsample_bytree::Float64 = 0.8
end

####################################################################################################

function loadDTparams(path::String)
  params = DecisionTreeParams()
  cfg = path != "" ? symbolise_keys(TOML.parsefile(path)["decisiontree"]) : Dict()
  return DecisionTreeParams(; merge(struct_to_dict(params), cfg)...)
end

####################################################################################################
