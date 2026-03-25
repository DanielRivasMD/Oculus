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
  for (k, v) in metrics
    if k == "ConfusionMatrix"
      println("Confusion Matrix:")
      for row in eachrow(v)
        println("  ", join(row, "  "))
      end
    else
      println("  $k: $v")
    end
  end
  return result
end

end
