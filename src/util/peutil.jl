####################################################################################################

module PECore

####################################################################################################

using DataFrames
using DelimitedFiles

####################################################################################################

export load_predictions

####################################################################################################

"""
    load_predictions(path::String) -> DataFrame

Read a CSV file with columns 'truth' and 'prediction' (both integer 0/1)
"""
function load_predictions(path::String)
  data, header = readdlm(path, ',', header = true)
  df = DataFrame(data, vec(header))
  for col in ["truth", "prediction"]
    col in names(df) || error("CSV must contain column '$col'")
  end
  return df
end

####################################################################################################

end

####################################################################################################
