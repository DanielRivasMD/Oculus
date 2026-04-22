####################################################################################################

module PECLI

####################################################################################################

using ArgParse
using Avicenna.Flow: Cache, launch
using ..PEFlow: flow

####################################################################################################

export run

####################################################################################################

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--in"
    help = "Input CSV file with columns 'truth' and 'prediction'"
    arg_type = String
    required = true
    "--no-cache"
    help = "Disable caching"
    action = :store_true
  end

  parsed = parse_args(args, s)

  config = Dict{String,Any}("infile" => parsed["in"])

  cache = Cache("cache/performance", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  println("Performance metrics: $(parsed["in"]):")
  metrics = result.stage_outputs["02_compute_metrics"]

  metric_order = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1Score",
    "BalancedAccuracy",
    "MCC",
    "FPR",
    "FNR",
    "FDR",
    "FOR",
    "NPV",
  ]

  for key in metric_order
    if haskey(metrics, key)
      println("  $key: $(metrics[key])")
    end
  end

  if haskey(metrics, "ConfusionMatrix")
    cm = metrics["ConfusionMatrix"]
    println("\nConfusion Matrix:")
    println("                Predicted")
    println("          Ancient    Modern")
    println("Actual")
    println("Ancient   $(cm[1,1])        $(cm[1,2])")
    println("Modern    $(cm[2,1])        $(cm[2,2])")
  end

  return result
end

####################################################################################################

end

####################################################################################################
