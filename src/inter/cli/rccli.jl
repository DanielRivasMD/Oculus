module RCCLI

using ArgParse
using Avicenna.Flow: Cache, launch
using ..RCFlow: flow

export run

function run(args)
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--single"
    help = "CSV with ground truth (columns: id,p0,p1,truth)"
    arg_type = String
    "--modern"
    help = "CSV for modern class (no truth column)"
    arg_type = String
    "--ancient"
    help = "CSV for ancient class (no truth column)"
    arg_type = String
    "--out"
    help = "Output HTML report"
    arg_type = String
    default = "roc_report.html"
    "--no-cache"
    help = "Disable caching"
    action = :store_true
    "--verbose"
    help = "Enable verbose diagnostics"
    action = :store_false
  end

  parsed = parse_args(args, s)

  # Build config dict
  config = Dict{String,Any}("out" => parsed["out"])
  if parsed["single"] != nothing
    config["single"] = parsed["single"]
  elseif parsed["modern"] != nothing && parsed["ancient"] != nothing
    config["modern"] = parsed["modern"]
    config["ancient"] = parsed["ancient"]
  else
    error("Either --single or both --modern AND --ancient must be provided")
  end

  cache = Cache("cache/roc", !parsed["no-cache"])
  result = launch(flow, config, cache = cache)

  if parsed["verbose"]
    println("ROC report saved to $(config["out"])")
  end
  return result
end

end
