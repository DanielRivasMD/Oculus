####################################################################################################

using TOML

####################################################################################################

"""
    loadParams(path, ::Type{T}; section=nothing, postprocess=nothing)

Generic TOML-backed loader for @with_kw parameter structs.

Arguments:
- `path::Union{Nothing,String}` — TOML file path or nothing
- `T` — the parameter struct type
- `section` — optional TOML section name (Symbol or String)
- `postprocess` — optional function(cfg::Dict) -> Dict for custom fixes

Returns:
- An instance of `T` with TOML overrides applied.
"""
function loadParams(
  path::Union{Nothing,String},
  ::Type{T};
  section = nothing,
  postprocess = nothing,
) where {T}

  defaults = struct_to_dict(T())

  # No config file → return defaults
  if path === nothing || path == ""
    return T(; defaults...)
  end

  raw = TOML.parsefile(path)

  # Extract section if provided
  cfg = section === nothing ? raw : raw[string(section)]
  cfg = symbolise_keys(cfg)

  # Optional postprocessing hook
  if postprocess !== nothing
    cfg = postprocess(cfg)
  end

  merged = merge(defaults, cfg)
  return T(; merged...)
end

####################################################################################################
