####################################################################################################

# DOC: kept for source code reference purpose, otherwise deprecated
using TOML
using FilePathsBase: basename, splitext, joinpath, dirname, isabspath, isdir, mkpath

####################################################################################################

function sysimage_args()
  desc = HELP * "Build a custom Julia sysimage for a script\n"

  s = ArgParseSettings(description = desc)

  @add_arg_table s begin
    "script"
    help = "Path to the Julia script to analyze"
    arg_type = String

    "--out"
    help = "Output sysimage path (default = <scriptname>.so)"
    default = nothing

    "--middleman"
    help = "Use an intermediary precompile driver file (default = false)"
    action = :store_true

    "--exclude"
    help = "Comma-separated list of packages to exclude"
    arg_type = String
    default = ""
  end

  args = parse_args(s)

  # Normalize exclude into a Vector{String}
  excl = isempty(args["exclude"]) ? String[] : split(args["exclude"], ',') .|> strip
  args["exclude"] = excl

  return args
end

####################################################################################################
