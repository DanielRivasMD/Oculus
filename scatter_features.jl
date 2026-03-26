using DataFrames
using DelimitedFiles
using Plots

function readdf(path; sep = ',')
  data, header = readdlm(path, sep, header = true)
  return DataFrame(data, vec(header))
end

df = readdf("data/cache/features/engineered.csv")

p1 = scatter(
  df.ga3p_5,
  df.ct5p_5,
  group = df.label,
  alpha = 0.3,
  xlabel = "ga3p_5",
  ylabel = "ct5p_5",
  title = "ga3p_5 vs ct5p_5 colored by label",
  legend = true,
  legendtitle = "Label",
  labels = ["Ancient (0)" "Modern (1)"],
)
savefig(p1, "data/graph/png/ga3p5_vs_ct5p5_all.png")

p2 = scatter(
  df.ga3p_5[df.label.==0],
  df.ct5p_5[df.label.==0],
  color = :blue,
  alpha = 0.3,
  xlabel = "ga3p_5",
  ylabel = "ct5p_5",
  title = "Ancient only",
  legend = false,
)
savefig(p2, "data/graph/png/ga3p5_vs_ct5p5_ancient.png")

p3 = scatter(
  df.ga3p_5[df.label.==1],
  df.ct5p_5[df.label.==1],
  color = :red,
  alpha = 0.3,
  xlabel = "ga3p_5",
  ylabel = "ct5p_5",
  title = "Modern only",
  legend = false,
)
savefig(p3, "data/graph/png/ga3p5_vs_ct5p5_modern.png")
