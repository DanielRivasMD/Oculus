#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "Oculus.jl"))
using .Oculus

Oculus.CLI.run_feature_extraction(ARGS)
