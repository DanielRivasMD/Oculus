####################################################################################################

using FreqTables

####################################################################################################

"""

    accuracy(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate accuracy from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> accuracy(x)
0.9064039408866995
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function accuracy(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return (a[1, 1] + a[2, 2]) / (sum(a))
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    balancedAccuracy(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate balanced accuracy from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> balancedAccuracy(x)
0.7883333333333333
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function balancedAccuracy(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return (sensitivity(a) + specificity(a)) / (2)
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    FDR(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate False Discovery Rate (FDR) from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> FDR(x)
0.9
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function FDR(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[1, 2] / (a[1, 1] + a[1, 2])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    FNR(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate False Negative Rate (FNR) from contingency table or confusion matrix.
Also called Miss Rate.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> FNR(x)
0.3333333333333333
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function FNR(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[2, 1] / (a[1, 1] + a[2, 1])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    FOR(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate False Omission Rate (FOR) from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> FOR(x)
0.00546448087431694
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function FOR(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[2, 1] / (a[2, 1] + a[2, 2])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    FPR(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate False Positive Rate (FPR) from contingency table or confusion matrix.
Also called Fall-out.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> FPR(x)
0.09
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function FPR(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[1, 2] / (a[1, 2] + a[2, 2])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    fScore(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate f1-score from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> fScore(x)
0.1739130434782609
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function fScore(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return 2 * ((PPV(a) * sensitivity(a)) / (PPV(a) + sensitivity(a)))
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    MCC(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate Mathews Correlation Coeficient (MCC) from contingency table or confusion matrix.
Also called Φ Coeficient.

# Examples
```jldoctest
julia> x = [6 1; 2 3]
julia> MCC(x)
0.47809144373375745

julia> x = [20 180; 10 1820]
julia> MCC(x)
0.23348550853492078
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function MCC(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return ((a[1, 1] * a[2, 2]) - (a[1, 2] * a[2, 1])) / (sqrt((a[1, 1] + a[1, 2]) * (a[1, 1] + a[2, 1]) * (a[2, 2] + a[1, 2]) * (a[2, 2] + a[2, 1])))
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    NPV(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate Negative Predictive Value (NPV) from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> NPV(x)
0.994535519125683
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function NPV(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[2, 2] / (a[2, 2] + a[2, 1])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    PPV(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate Positive Predictive Value (PPV) from contingency table or confusion matrix.
Also called Precision.

# Examples
```jldoctest
julia> x = [20 180; 10 1820]
julia> PPV(x)
0.1
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function PPV(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[1, 1] / (a[1, 1] + a[1, 2])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    sensitivity(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate Sensitivity from contingency table or confusion matrix.
Also called True Positive Rate (TPR), or Recall.

# Examples
```jldoctest
julia> x = [10 40; 5 45]
julia> sensitivity(x)
sensitivity = 0.6666666666666666

julia> x = [20 33; 10 37]
julia> sensitivity(x)
sensitivity = 0.6666666666666666

julia> x = [20 180; 10 1820]
julia> sensitivity(x)
sensitivity = 0.6666666666666666
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function sensitivity(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[1, 1] / (a[1, 1] + a[2, 1])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    specificity(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate Specificity from contingency table or confusion matrix.
Also called True Negative Rate (TPR), or Selectivity.

# Examples
```jldoctest
julia> x = [10 40; 5 45]
julia> specificity(x)
specificity = 0.5294117647058824

julia> x = [20 33; 10 37]
julia> specificity(x)
specificity = 0.5285714285714286

julia> x = [20 180; 10 1820]
julia> specificity(x)
specificity = 0.91
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function specificity(a::M) where M <: Matrix{N} where N <: Number
  if size(a) == (2, 2)
    return a[2, 2] / (a[2, 2] + a[1, 2])
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"""

    performance(tbVc::V, labelVc::V)
      where V <: Vector{N}
      where N <: Number

# Description
Calculate performance from prediction vector and supervised vector.


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function performance(tbVc::V, labelVc::V) where V <: Vector{N} where N <: Number

  # declare internal copy
  tbVec = copy(tbVc)

  # reassign frecuency labels
  labels = [1, 2]
  tbVec[tbVec .> 1] .= 2

  return performance(adjustFq(tbVec, labelVc, labels))
end

####################################################################################################

"""

    performance(a::M)
      where M <: Matrix{N}
      where N <: Number

# Description
Calculate performance from contingency table or confusion matrix.

# Examples
```jldoctest
julia> x = [10 40; 5 45]
julia> performance(x)
(sensitivity = 0.6666666666666666, specificity = 0.5294117647058824, accuracy = 0.55, fScore = 0.30769230769230765, PPV = 0.2, NPV = 0.9, FPR = 0.47058823529411764, FNR = 0.3333333333333333, FDR = 0.8, FOR = 0.1, MCC = 0.14002800840280097)

julia> x = [20 33; 10 37]
julia> performance(x)
(sensitivity = 0.6666666666666666, specificity = 0.5285714285714286, accuracy = 0.57, fScore = 0.4819277108433735, PPV = 0.37735849056603776, NPV = 0.7872340425531915, FPR = 0.4714285714285714, FNR = 0.3333333333333333, FDR = 0.6226415094339622, FOR = 0.2127659574468085, MCC = 0.17926163185860888)

julia> x = [20 180; 10 1820]
julia> performance(x)
(sensitivity = 0.6666666666666666, specificity = 0.91, accuracy = 0.9064039408866995, fScore = 0.1739130434782609, PPV = 0.1, NPV = 0.994535519125683, FPR = 0.09, FNR = 0.3333333333333333, FDR = 0.9, FOR = 0.00546448087431694, MCC = 0.23348550853492078)
```


See also: [`performance`](@ref), [`accuracy`](@ref), [`balancedAccuracy`](@ref), [`fScore`](@ref), [`sensitivity`](@ref), [`specificity`](@ref), [`PPV`](@ref), [`NPV`](@ref), [`FPR`](@ref), [`FNR`](@ref), [`FDR`](@ref), [`FOR`](@ref), [`MCC`](@ref).
"""
function performance(a::M) where M <: Matrix{N} where N <: Number
  @info a
  if size(a) == (2, 2)
    return Dict(
      "Sensitivity" => sensitivity(a),
      "Specificity" => specificity(a),
      "Accuracy" => accuracy(a),
      "BalancedAccuracy" => balancedAccuracy(a),
      "FScore" => fScore(a),
      "PPV" => PPV(a),
      "NPV" => NPV(a),
      "FPR" => FPR(a),
      "FNR" => FNR(a),
      "FDR" => FDR(a),
      "FOR" => FOR(a),
      "MCC" => MCC(a),
    )
  else
    @error "Array does not have the proper size"
  end
end

####################################################################################################
####################################################################################################

"transform freqtable => dataframe"
function convertFqDf(fq; colnames = ["Value", "Frecuency"])
  return DataFrames.DataFrame([names(fq)[1] fq.array], colnames)
end


"transform freqtable => dataframe template"
function convertFqDf(fq, templ; colnames = ["Value", "Frecuency"])

  fq = convertFqDf(fq)

  out = DataFrames.DataFrame([templ zeros(Int64, length(templ))], colnames)

  for i ∈ axes(fq, 1)
    out[findall(fq[i, 1] .== out[:, 1]), 2] .= fq[i, 2]
  end

  return out
end

####################################################################################################

"adjust & concatenate frecuency tables"
function adjustFq(tbVec, labelVc, labels)
  positives = tbVec[labelVc[:, 1] .== 1] |> freqtable |> p -> convertFqDf(p, labels) |> p -> sort(p, rev = true)
  negatives = tbVec[labelVc[:, 1] .== 0] |> freqtable |> p -> convertFqDf(p, labels) |> p -> sort(p, rev = true)
  return [positives[:, 2] negatives[:, 2]]
end

####################################################################################################
