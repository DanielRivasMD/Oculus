module DADoc

using Markdown
using Weave
using Avicenna.Flow: Result
using Base.Filesystem: mktempdir
using DataFrames
using DelimitedFiles

export report_deamination

"""
    report_deamination(result::Result) -> String

Generate a Markdown summary of the deamination analysis result.
"""
function report_deamination(result::Result)
  # Retrieve the CSV path from the config
  csv_path = result.origin["config"]["csv"]
  if !isfile(csv_path)
    return "No CSV file found at $csv_path"
  end

  # Read the CSV to get basic stats
  data = readdlm(csv_path, ',', String)
  header = data[1, :]
  rows = data[2:end, :]
  n_positions = size(rows, 1)

  # Extract modern and ancient labels from header
  modern_name = replace(header[2], r"_A$" => "")
  ancient_name = replace(header[6], r"_A$" => "")

  md = """
    # Deamination Analysis Report

    ## Input files
    - Modern: `$(result.origin["config"]["modern"])`
    - Ancient: `$(result.origin["config"]["ancient"])`

    ## Output
    - CSV: `$csv_path`
    - Plot: `$(result.origin["config"]["png"])`

    ## Positions analysed
    $n_positions positions (length of sequences)

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
report_html(result::Result; outpath=nothing) -> String

Generate an HTML report from the deamination analysis result.
"""
function report_html(result::Result; outpath = nothing)
  md = report_deamination(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.html") : outpath
  weave(infile; doctype = "md2html", out_path = outfile)
  return outfile
end

"""
report_pdf(result::Result; outpath=nothing) -> String

Generate a PDF report from the deamination analysis result.
"""
function report_pdf(result::Result; outpath = nothing)
  md = report_deamination(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.pdf") : outpath
  weave(infile; doctype = "md2pdf", out_path = outfile)
  return outfile
end

end
