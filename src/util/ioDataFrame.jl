###################################################################################################

using DataFrames
using DelimitedFiles

###################################################################################################

"read dataframe"
function readdf(path; sep = '\t')
  data, header = readdlm(path, sep, header = true)
  return DataFrame(data, vec(header))
end

###################################################################################################

"write dataframe"
function writedf(path, df::DataFrame; sep = ',')
  header = permutedims(names(df))  # 1×N matrix of strings
  data = Matrix(df)
  writedlm(path, vcat(header, data), sep)
end

###################################################################################################

