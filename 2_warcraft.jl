### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f18ceb3e-accd-11ef-3897-6b9be51b1868
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
end

# ╔═╡ 5bbc6df8-1111-4311-ae26-d534cd1af8ff
begin
	using PlutoUI: Slider
end

# ╔═╡ 9b2c1885-ba66-48ff-8c18-21cf7e75d1e6
using DecisionFocusedLearningBenchmarks

# ╔═╡ afea8933-e980-4431-930a-45c37b870897
b = WarcraftBenchmark()

# ╔═╡ 6dad0626-7a84-4f83-8fc5-f447a1dd337a
dataset_size = 100

# ╔═╡ ce0bd9ec-505a-447a-8d28-5527c9bb3a7a
dataset = generate_dataset(b, dataset_size)

# ╔═╡ 81d0c2b8-993d-4074-8db8-bff9637fedb9
typeof(dataset)

# ╔═╡ c14fe6de-0ab3-4707-b170-17b229679320
md"index = $(@bind index Slider(1:100; show_value=true))"

# ╔═╡ 49997eb9-c0e7-4dd7-88e0-15bc3d98caa7
plot_data(b, dataset[index])

# ╔═╡ Cell order:
# ╠═f18ceb3e-accd-11ef-3897-6b9be51b1868
# ╠═5bbc6df8-1111-4311-ae26-d534cd1af8ff
# ╠═9b2c1885-ba66-48ff-8c18-21cf7e75d1e6
# ╠═afea8933-e980-4431-930a-45c37b870897
# ╠═6dad0626-7a84-4f83-8fc5-f447a1dd337a
# ╠═ce0bd9ec-505a-447a-8d28-5527c9bb3a7a
# ╠═81d0c2b8-993d-4074-8db8-bff9637fedb9
# ╟─c14fe6de-0ab3-4707-b170-17b229679320
# ╠═49997eb9-c0e7-4dd7-88e0-15bc3d98caa7
