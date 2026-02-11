####################################################################################################
# cli args
####################################################################################################

begin
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.UTIL, "args.jl"))
end

# Parse CLI arguments
args = regression_args()

####################################################################################################
# Imports
####################################################################################################

using DataFrames
using DelimitedFiles
using GLM
using StatsModels
using FilePathsBase: basename, splitext

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(Paths.UTIL, "ioDataFrame.jl"))
end;

####################################################################################################
# Main execution
####################################################################################################

if !isinteractive() && PROGRAM_FILE !== nothing

  infile  = args["in"]
  outfile = args["out"]

  println("Loading dataframe from $infile")
  df = readdf(infile; sep=',')
  rename!(df, Symbol.(names(df)))

  # # Ensure label exists
  # @assert :label in names(df) "DataFrame must contain a :label column"

  # Separate features and label
  feature_cols = setdiff(names(df), [string(:label)])

  # Build formula: label ~ x1 + x2 + ...
  f = Term(:label) ~ sum(Term.(Symbol.(feature_cols)))

  println("Fitting linear regression model...")
  model = lm(f, df)

  println("\nModel coefficients:")
  println(coef(model))

  # Optionally write predictions
  if outfile !== nothing
    preds = predict(model)
    outdf = DataFrame(pred = preds)
    writedf(outfile, outdf; sep=',')
    println("Predictions written to $outfile")
  end
end

####################################################################################################
