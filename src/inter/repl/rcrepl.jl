module RCREPL

using Avicenna.Flow: Cache, launch
using ..RCFlow: flow

export run

"""
    run(; out="roc_report.html", no_cache=false, kwargs...)

REPL‑friendly entry point for ROC analysis.

Keyword arguments (exactly one of the following must be supplied):
- `single::String`   Path to single CSV with ground truth
- `modern::String`   Path to modern predictions CSV
- `ancient::String`  Path to ancient predictions CSV
"""
function run(; out::String = "roc_report.html", no_cache::Bool = false, kwargs...)
  config = Dict{String,Any}("out" => out)
  for (k, v) in kwargs
    if k in (:single, :modern, :ancient)
      config[string(k)] = v
    end
  end
  if !haskey(config, "single") && !(haskey(config, "modern") && haskey(config, "ancient"))
    error("Provide either single or modern+ancient")
  end
  cache = Cache("cache/roc", !no_cache)
  return launch(flow, config, cache = cache)
end

end
