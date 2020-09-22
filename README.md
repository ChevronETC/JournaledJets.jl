# JournaledJets.jl

| **Documentation** | **Action Statuses** |
|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][doc-build-status-img]][doc-build-status-url] [![][build-status-img]][build-status-url] [![][code-coverage-img]][code-coverage-results] |

This package contains distributed block operators and vectors for Jets.jl.  It
builds on top of the block operators in Jets.jl, providing a parallel distributed, fault tolerant version of block operators and block vectors that are used to orchestrate
distributed (in-memory) storage and compute.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://chevronetc.github.io/JournaledJets.jl/dev/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ChevronETC.github.io/JournaledJets.jl/stable

[doc-build-status-img]: https://github.com/ChevronETC/JournaledJets.jl/workflows/Documentation/badge.svg
[doc-build-status-url]: https://github.com/ChevronETC/JournaledJets.jl/actions?query=workflow%3ADocumentation

[build-status-img]: https://github.com/ChevronETC/JournaledJets.jl/workflows/Tests/badge.svg
[build-status-url]: https://github.com/ChevronETC/JournaledJets.jl/actions?query=workflow%3A"Tests"