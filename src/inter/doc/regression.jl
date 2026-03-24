module RGDoc

using Markdown
using Weave
using Avicenna.Flow: Result
using Base.Filesystem: mktempdir

export report_regression

"""
    report_regression(result::Result) -> String

Generate a Markdown summary of the regression result.
"""
function report_regression(result::Result)
  eval_metrics = result.stage_outputs["evaluate"]
  if isempty(eval_metrics)
    return "No evaluation metrics available (test set empty?)."
  end
  cm = eval_metrics["confusion_matrix"]
  md = """
    # Regression Analysis Report

    ## Configuration
    $(result.origin["config"])

## Evaluation Metrics
- Accuracy: $(eval_metrics["accuracy"])
- Sensitivity (Recall): $(eval_metrics["sensitivity"])
- Specificity: $(eval_metrics["specificity"])
- Precision: $(eval_metrics["precision"])
- F1 Score: $(eval_metrics["f1"])

## Confusion Matrix
$(cm)

## Origin
- Workflow: $(result.origin["workflow"])
- Version: $(result.origin["version"])
- Time: $(result.origin["timestamp"])
- Cache hits: $(join(result.origin["cache_hits"], ", "))
"""
  return md
end

"""
report_html(result::Result; outpath=nothing) -> String

Generate an HTML report from the regression result.
"""
function report_html(result::Result; outpath = nothing)
  md = report_regression(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.html") : outpath
  weave(infile; doctype = "md2html", out_path = outfile)
  return outfile
end

"""
report_pdf(result::Result; outpath=nothing) -> String

Generate a PDF report from the regression result.
"""
function report_pdf(result::Result; outpath = nothing)
  md = report_regression(result)
  tmpdir = mktempdir()
  infile = joinpath(tmpdir, "report.jmd")
  write(infile, md)
  outfile = isnothing(outpath) ? joinpath(tmpdir, "report.pdf") : outpath
  weave(infile; doctype = "md2pdf", out_path = outfile)
  return outfile
end

end
