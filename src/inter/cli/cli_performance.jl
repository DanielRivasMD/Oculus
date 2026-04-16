module PECLI

using ArgParse
using Avicenna.Flow: Cache, run
using ..PEFlow: performance_flow

export run_performance

function run_performance(args)
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
  result = run(performance_flow, config, cache = cache)

  println("Performance metrics for $(parsed["in"]):")
  metrics = result.stage_outputs["compute_metrics"]

  # Define the order for numeric metrics
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

  # Print numeric metrics in the defined order
  for key in metric_order
    if haskey(metrics, key)
      println("  $key: $(metrics[key])")
    end
  end

  # Print the confusion matrix last
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

end
