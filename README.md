<!-- Title -->
<h1 align="center">
  Oculus
</h1>

<!-- description -->
<p align="center">
  <strong>Bringing Ancient DNA into Focus</strong>
</p>

<!-- Information badges -->
<p align="center">
  <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/active.svg?style=flat-square" />
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <a href="https://DanielRivasMD.github.io/Oculus/stable">
    <img alt="Stable" src="https://img.shields.io/badge/docs-stable-blue.svg">
  </a>
  </a>
  <a href="https://DanielRivasMD.github.io/Oculus/dev">
    <img alt="Dev" src="https://img.shields.io/badge/docs-dev-blue.svg">
  </a>
</p>

<!-- Community -->
<p align="center">
  <a href="https://github.com/DanielRivasMD/Oculus/discussions">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
</p>

<!-- Version and documentation badges -->
<p align="center">
  <a href="https://github.com/DanielRivasMD/Oculus/releases">
    <img alt="GitHub tag (latest SemVer pre-release)" src="https://img.shields.io/github/v/tag/DanielRivasMD/MindReader.jl?include_prereleases&label=latest%20version&logo=github&sort=semver&style=flat-square">
  </a>
</p>

<!-- ![Oculus](assets/Oculus.png) -->

## Overview
Oculus is a command line tool for identifying Ancient DNA sequences


## Features
WIP: elaborate on app features


## Installation

It is recommend to install `julia` through `juliaup`. For additional information, visit their official website.
To enter the julia REPL, call `julia` in your project path.
From the julia REPL, activate the local enviroment and install the dependencies by entering the package manager, as follows:

```julia
julia>]
pkg> activate .
pkg> instantiate
```

To exit Pkg mode, just backspace.

For more information, see the [Pkg documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/).


## Usage

Oculus provides executable scripts at `src/bin`, which can be called from the command line:

```bash
julia --project src/bin/cnn.jl --help
julia --project src/bin/performance.jl --help
julia --project src/bin/inference.jl --help
```

The expected data path structure (shown below) will be created and checked at the start of any run.
Thus, running `help` will trigger the path creation.

```
Oculus/
├── data/
│   ├── bam/
│   ├── fasta/
│   └── inference/
├── graph/
│   └── performance/
└── model/
```

By using `Parameters.jl` and `CNNParams` / `SampleParams` structs the user can control all of the parameters of the model in a user friendly way.
These structs can also be bound to `toml` files at the command line.


## Example

It is recommended to perform a dummy run prior to train a full CNN.
This can be done by preparing a dummy sample with a few sequences, e.g., 100.
Place the sequences at `data/fasta`: `French_37nt_head.fasta` & `Neandertal_37nt_head.fasta`.
Call the CNN trainer:

```bash
julia --project src/bin/cnn.jl
```

Note that this specific names and parameters are default, hence, no further configs are required.
To train a CNN with other settings, add them to the call:

```bash
julia --project src/bin/cnn.jl --sample toml/sample75nt.toml --cnn toml/cnn1l_wobn.toml
```

Likewise, once a model is created, their performance can be assessed, or inferences can be made calling the appropiate executable scripts.


## License
Copyright (c) 2025

## License

This package is licensed under the MIT Expat license. See [LICENSE](LICENSE) for more informaiton.

---

**Author's Note**: This package is still under active development and is subject to change.

