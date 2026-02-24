module FeatureWorkflow

using Avicenna.Workflow
using ..Process

using DataFrames
using DelimitedFiles

function writedf(path, df::DataFrame; sep = ',')
  header = permutedims(names(df))  # 1×N matrix of strings
  data = Matrix(df)
  writedlm(path, vcat(header, data), sep)
end


const feature_workflow = WorkflowConfig(
  "feature_extraction",
  [
    Stage("load_modern", (config, _) -> Process.load_fasta(config["modern"]), "1.0"),
    Stage("load_ancient", (config, _) -> Process.load_fasta(config["ancient"]), "1.0"),
    Stage(
      "build_modern_df",
      (config, prev) ->
        Process.build_df(prev["load_modern"], 1; onehot = config["onehot"]),
      "1.0",
    ),
    Stage(
      "build_ancient_df",
      (config, prev) ->
        Process.build_df(prev["load_ancient"], 0; onehot = config["onehot"]),
      "1.0",
    ),
    Stage(
      "merge",
      (_, prev) -> vcat(prev["build_modern_df"], prev["build_ancient_df"]),
      "1.0",
    ),
    Stage(
      "write",
      (config, prev) -> writedf(config["out"], prev["merge"]; sep = ","),
      "1.0",
    ),
  ],
  "1.0",
)

export feature_workflow

end
