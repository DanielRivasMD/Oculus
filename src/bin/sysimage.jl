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
  # Load path definitions
  include(joinpath(PROGRAM_FILE === nothing ? "src" : "..", "config", "paths.jl"))
  using .Paths
  Paths.ensure_dirs()

  # Load configuration structs
  include(joinpath(Paths.CONFIG, "args.jl"))      # Args API (now includes infer_args)
end;

####################################################################################################

"""
    build_sysimage(script::AbstractString; sysimage_path=nothing)

Read a Julia script, extract its `using`/`import` dependencies,
and build a sysimage named after the script (without extension).
"""
function build_sysimage(script::AbstractString; sysimage_path=nothing)
    # Read script lines
    lines = readlines(script)

    # Regex to capture package names from `using`, `import`, or `@eval using`
    pat = r"^(?:@eval\s+)?(?:using|import)\s+([A-Za-z0-9_.]+)"

    pkgs = String[]
    for line in lines
        m = match(pat, strip(line))
        if m !== nothing
            raw = m.captures[1]

            # Skip relative modules (e.g. `.Paths`)
            if startswith(raw, ".")
                continue
            end

            # Take the top-level package (before any '.')
            top = split(raw, '.')[1]

            # Only push valid identifiers
            if !isempty(top) && occursin(r"^[A-Za-z]\w*$", top)
                push!(pkgs, top)
            end
        end
    end
    pkgs = unique(pkgs)

    # Default sysimage path goes into Paths.SYSIMAGE
    scriptname = splitext(basename(script))[1]
    default_path = joinpath(Paths.SYSIMAGE, scriptname * ".so")
    sysimage_path = isnothing(sysimage_path) ? default_path : sysimage_path

    # Write temporary driver that just imports the packages
    tmpfile = joinpath(pwd(), "precompile_driver_" * string(uuid4()) * ".jl")
    open(tmpfile, "w") do io
        for pkg in pkgs
            println(io, "using $pkg")
        end
    end

    println("Building sysimage $sysimage_path with packages: ", pkgs)

    create_sysimage(Symbol.(pkgs);
        sysimage_path=sysimage_path,
        precompile_execution_file=tmpfile
    )

    rm(tmpfile; force=true)
    println("Sysimage written to $sysimage_path")
end

####################################################################################################

# CLI entrypoint
if abspath(PROGRAM_FILE) == @__FILE__
    args = sysimage_args()
    build_sysimage(args["script"]; sysimage_path=args["out"])
end

####################################################################################################
