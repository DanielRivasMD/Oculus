####################################################################################################

"Define function to split data (source: Huda Nassar)"
function perClassSplits(vec, percent)
  uniq_class = unique(vec)
  keep_index = []
  for class in uniq_class
    class_index = findall(vec .== class)
    row_index = randsubseq(class_index, percent)
    push!(keep_index, row_index...)
  end
  return keep_index
end

####################################################################################################
