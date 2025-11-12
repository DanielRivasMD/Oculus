using DelimitedFiles
using UnicodePlots

# File paths
modern  = "French_chr20_75nt_cnn1l_sample_chr1_75nt_f1_2025-11-11_150447.csv"
ancient = "Neandertal_chr20_75nt_cnn1l_sample_chr1_75nt_f1_2025-11-11_150447.csv"

# Load CSVs with header
modern_data  = readdlm(modern, ',', header=true)
ancient_data = readdlm(ancient, ',', header=true)

# Extract the actual data matrix
modern_mat  = modern_data[1]
ancient_mat = ancient_data[1]

# Columns: 1=id, 2=p0, 3=p1
modern_p1  = Float32.(modern_mat[:,3])
ancient_p1 = Float32.(ancient_mat[:,3])

println("Distribution of predicted probability for class=1 (Neandertal):")

# Separate histograms
histogram(modern_p1; nbins=20, title="Modern (French)", xlabel="p1", ylabel="count") |> display
histogram(ancient_p1; nbins=20, title="Ancient (Neandertal)", xlabel="p1", ylabel="count") |> display

# Side-by-side comparison using barplot of summary stats
using Statistics
means = [mean(modern_p1), mean(ancient_p1)]
barplot(["Modern","Ancient"], means; title="Mean p1", xlabel="Group", ylabel="Mean probability") |> display
