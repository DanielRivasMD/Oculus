####################################################################################################

function confusion_matrix(y_true::Vector{Int}, y_pred::Vector{Int}; classes = nothing)
  # If classes not provided, infer them from the data
  if classes === nothing
    classes = sort(unique(vcat(y_true, y_pred)))
  end

  n = length(classes)
  cm = zeros(Int, n, n)

  # Map class value → matrix index
  idxmap = Dict(c => i for (i, c) in enumerate(classes))

  for (t, p) in zip(y_true, y_pred)
    cm[idxmap[t], idxmap[p]] += 1
  end

  return cm, classes
end

####################################################################################################
