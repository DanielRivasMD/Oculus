#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "Performance.jl"))
using .PE

PECLI.run_performance(ARGS)
