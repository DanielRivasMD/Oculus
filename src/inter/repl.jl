module REPL

using Avicenna.Workflow
using ..FeatureWorkflow

function run_features(modern, ancient; onehot = false, out = "features.csv")
  config = Dict("modern" => modern, "ancient" => ancient, "onehot" => onehot, "out" => out)
  Workflow.run(feature_workflow, config)
end

end
