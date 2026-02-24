module CLI

using Avicenna.Workflow
using ..FeatureWorkflow

function run_feature_extraction(args)
  @info args
  config = Dict(
    "modern" => args["modern"],
    "ancient" => args["ancient"],
    "onehot" => args["onehot"],
    "out" => args["out"],
  )

  Workflow.run(feature_workflow, config)
end

end
