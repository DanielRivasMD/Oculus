####################################################################################################

module RCFlow

####################################################################################################

using Avicenna.Flow: Stage, Config
using ..RCCore

####################################################################################################

export flow

####################################################################################################

const flow = Config(
  "roc_analysis",
  [
    Stage(
      "01_load_data",
      (config, _) -> begin
        if haskey(config, "single")
          probs, labels = RCCore.load_probs_labels(config["single"]; has_truth = true)
          title = "ROC (single file)"
        elseif haskey(config, "modern") && haskey(config, "ancient")
          modern_probs, _ = RCCore.load_probs_labels(config["modern"]; label = 0)
          ancient_probs, _ = RCCore.load_probs_labels(config["ancient"]; label = 1)
          probs = vcat(modern_probs, ancient_probs)
          labels =
            vcat(zeros(Int, length(modern_probs)), ones(Int, length(ancient_probs)))
          title = "ROC (modern vs ancient)"
        else
          error("Provide --single OR both --modern and --ancient")
        end
        return (probs = probs, labels = labels, title = title)
      end,
      "1.0",
    ),
    Stage(
      "02_compute_roc",
      (_, prev) -> begin
        fpr, tpr =
          RCCore.roc_curve(prev["01_load_data"].probs, prev["01_load_data"].labels)
        return (fpr = fpr, tpr = tpr)
      end,
      "1.0",
    ),
    Stage(
      "03_report",
      (config, prev) -> begin
        title = prev["01_load_data"].title
        fpr = prev["02_compute_roc"].fpr
        tpr = prev["02_compute_roc"].tpr
        RCCore.generate_roc_report(fpr, tpr, title, config["out"])
        return nothing
      end,
      "1.0",
    ),
  ],
  "1.0",
)

####################################################################################################

end

####################################################################################################
