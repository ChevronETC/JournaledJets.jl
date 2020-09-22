push!(LOAD_PATH,"../src/")

using Documenter, JournaledJets

makedocs(
    sitename="JournaledJets.jl",
    modules=[JournaledJets],
    pages = [
        "Home" => "index.md",
        "reference.md",
        ]
)

deploydocs(
    repo = "github.com/ChevronETC/JournaledJets.jl"
)
