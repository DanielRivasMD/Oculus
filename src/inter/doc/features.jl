####################################################################################################

module FEDoc

####################################################################################################

using DataFrames
using Markdown
using Weave
using Avicenna.Flow: Result
using Base.Filesystem: mktempdir

####################################################################################################

export report_markdown, report_html, report_pdf

####################################################################################################

"""
    report_feature(result::Result) -> String

Generate a Markdown summary of the feature extraction result.
"""
function report_markdown(result::Result)
  df = result.stage_outputs["merge"]
  modern_count = count(==(1), df.label)
  ancient_count = count(==(0), df.label)
  feature_cols = filter(c -> c != :label, names(df))

  md = """
    # Feature Extraction Report

    ## Dataset
    - Modern sequences: $modern_count
    - Ancient sequences: $ancient_count
    - Total sequences: $(nrow(df))

    ## Features
    - Number of features: $(length(feature_cols))
    - Feature names: `$(join(feature_cols, ", "))`

    ## Origin
    - Workflow: $(result.origin["workflow"])
    - Version: $(result.origin["version"])
    - Time: $(result.origin["timestamp"])
    - Cache hits: $(join(result.origin["cache_hits"], ", "))

    ## Configuration
$(result.origin["config"])
"""
  return md
end

"""
report_html(result::Result; outpath=nothing)

Generate an HTML report from the feature extraction result.
"""
function report_html(result::Result; outpath = nothing)
  md = report_feature(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.html") : outpath
  weave(infile; doctype = "md2html", out_path = outfile)
  return outfile
end

"""
report_pdf(result::Result; outpath=nothing)

Generate a PDF report from the feature extraction result.
"""
function report_pdf(result::Result; outpath = nothing)
  md = report_feature(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.pdf") : outpath
  weave(infile; doctype = "md2pdf", out_path = outfile)
  return outfile
end

####################################################################################################

end

####################################################################################################
