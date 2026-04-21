####################################################################################################

module DAFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..DACore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "deamination_analysis",
  [
    Stage("01_load_ancient", (config, _) -> DACore.load_fasta(config["ancient"]), "1.0"),
    Stage("01_load_modern", (config, _) -> DACore.load_fasta(config["modern"]), "1.0"),
    Stage(
      "02_compute_composition",
      (config, prev) -> begin
        comp_ancient = DACore.position_composition(prev["01_load_ancient"])
        comp_modern = DACore.position_composition(prev["01_load_modern"])
        return (comp_modern, comp_ancient)
      end,
      "1.0",
    ),
    Stage(
      "03_write_csv",
      (config, prev) -> DACore.write_csv(
        config["csv"],
        config["ancient_name"],
        config["modern_name"],
        prev["02_compute_composition"][1],
        prev["02_compute_composition"][2],
      ),
      "1.0",
    ),
    Stage(
      "04_plot",
      (config, prev) -> DACore.plot_composition(config["csv"]; outfile = config["png"]),
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
