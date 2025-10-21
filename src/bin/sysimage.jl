####################################################################################################
# Imports
####################################################################################################

using PackageCompiler
using ArgParse
using UUIDs

####################################################################################################
# Load configuration
####################################################################################################

begin
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  include(joinpath(Paths.CONFIG, "args.jl"))
end

####################################################################################################

"""
    build_sysimage(script::AbstractString; sysimage_path=nothing, middleman::Bool=false, exclude=String[])

Read a Julia script, extract its `using`/`import` dependencies,
and build a sysimage named after the script (without extension).

- If `middleman=true`, generate a temporary driver file that imports the
  dependencies and feed it to PackageCompiler. Otherwise, build directly.
- If `exclude` is provided, those packages are removed from the build.
"""
function build_sysimage(script::AbstractString; sysimage_path=nothing,
                        middleman::Bool=false, exclude::Vector{String}=String[])
    lines = readlines(script)
    pat = r"^(?:@eval\s+)?(?:using|import)\s+([A-Za-z0-9_.]+)"

    pkgs = String[]
    for line in lines
        m = match(pat, strip(line))
        if m !== nothing
            raw = m.captures[1]
            if startswith(raw, ".")
                continue
            end
            top = split(raw, '.')[1]
            if !isempty(top) && occursin(r"^[A-Za-z]\w*$", top)
                push!(pkgs, top)
            end
        end
    end
    pkgs = unique(pkgs)

    # Apply exclusions
    if !isempty(exclude)
        pkgs = setdiff(pkgs, exclude)
    end

    scriptname = splitext(basename(script))[1]
    default_path = joinpath(Paths.SYSIMAGE, scriptname * ".so")
    sysimage_path = isnothing(sysimage_path) ? default_path : sysimage_path

    println("Building sysimage $sysimage_path with packages: ", pkgs)

    if middleman
        tmpfile = joinpath(pwd(), "precompile_driver_" * string(uuid4()) * ".jl")
        open(tmpfile, "w") do io
            for pkg in pkgs
                println(io, "using $pkg")
            end
        end

        create_sysimage(Symbol.(pkgs);
            sysimage_path=sysimage_path,
            precompile_execution_file=tmpfile
        )

        rm(tmpfile; force=true)
    else
        create_sysimage(Symbol.(pkgs);
            sysimage_path=sysimage_path
        )
    end

    println("Sysimage written to $sysimage_path")
end

####################################################################################################

# CLI entrypoint
if abspath(PROGRAM_FILE) == @__FILE__
    args = sysimage_args()
    build_sysimage(args["script"];
        sysimage_path=args["out"],
        middleman=args["middleman"],
        exclude=args["exclude"])
end

####################################################################################################
