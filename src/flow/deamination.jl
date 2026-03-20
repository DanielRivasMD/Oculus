module DAFlow

using Avicenna.Flow: Stage, Config
using ..DACore

export deamination_flow

const deamination_flow = Config(
  "deamination_analysis",
  [
    Stage(
      "load_modern",
      (config, _) -> DACore.load_fasta(config["modern"]),
      "1.0",
    ),
    Stage(
      "load_ancient",
      (config, _) -> DACore.load_fasta(config["ancient"]),
      "1.0",
    ),
    Stage(
      "compute_composition",
      (config, prev) -> (
        comp_modern = DACore.position_composition(prev["load_modern"]),
        comp_ancient = DACore.position_composition(prev["load_ancient"]),
        (comp_modern, comp_ancient),
      ),
      "1.0",
    ),
    Stage(
      "write_csv",
      (config, prev) -> DACore.write_csv(
        config["csv"],
        config["modern_name"],
        config["ancient_name"],
        prev["compute_composition"][1],
        prev["compute_composition"][2],
      ),
      "1.0",
    ),
    Stage(
      "plot",
      (config, prev) ->
        DACore.plot_composition(config["csv"]; outfile = config["png"]),
      "1.0",
    ),
  ],
  "1.0",
)

end
