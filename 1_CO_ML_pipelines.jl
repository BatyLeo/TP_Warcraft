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

# ╔═╡ 87660d1a-acca-11ef-02da-d113baade226
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
end

# ╔═╡ 5a34ecaa-55ca-4c0e-b4eb-59bae1b1c926
begin
	using Colors: Colors
	using LaTeXStrings: @L_str
	using LinearAlgebra: dot, norm
	using Markdown: MD, Admonition, Code
	using Plots: Plots, plot, plot!, scatter, scatter!, contour
	using PlutoTeachingTools: ChooseDisplayMode, tip, question_box, TwoColumn
	using PlutoUI: TableOfContents, Slider, CheckBox, Select
	using Zygote
end

# ╔═╡ edc13965-3c29-4ddd-ad7e-d1f5b4b9e6c9
using InferOpt

# ╔═╡ f608b9d7-2266-4686-8241-d0aa14501103
using DecisionFocusedLearningBenchmarks

# ╔═╡ 481c897f-0920-4c43-854c-49af36405cab
TableOfContents(depth=3)

# ╔═╡ 32eba97b-e192-429f-86a6-816545c75761
begin
	info(text; title="Info") = MD(Admonition("info", title, [text]));
	logocolors = Colors.JULIA_LOGO_COLORS;
    function get_angle(v)
        @assert !(norm(v) ≈ 0)
        v = v ./ norm(v)
        if v[2] >= 0
            return acos(v[1])
        else
            return π + acos(-v[1])
        end
    end

    function init_plot(title)
        pl = plot(;
            aspect_ratio=:equal,
            legend=:outerleft,
            xlim=(-1.1, 1.1),
            ylim=(-1.1, 1.1),
            title=title,
        )
        return pl
    end

    function plot_polytope!(pl, vertices)
        plot!(
            vcat(map(first, vertices), first(vertices[1])),
            vcat(map(last, vertices), last(vertices[1]));
            fillrange=0,
            fillcolor=:gray,
            fillalpha=0.2,
            linecolor=:black,
            label=L"\mathrm{conv}(\mathcal{V})"
        )
    end

    function plot_objective!(pl, θ)
        plot!(
            pl,
            [0.0, θ[1]],
            [0.0, θ[2]],
            color=logocolors.purple,
            arrow=true,
            lw=2,
            label=nothing
        )
        Plots.annotate!(
            pl,
            [-0.2 * θ[1]],
            [-0.2 * θ[2]],
            [L"\theta"],
        )
        return θ
    end

    function plot_maximizer!(pl, θ, polytope, maximizer)
        ŷ = maximizer(θ; polytope)
        scatter!(
            pl,
            [ŷ[1]],
            [ŷ[2]];
            color=logocolors.red,
            markersize=9,
            markershape=:square,
            label=L"f(\theta)"
        )
    end

    function plot_distribution!(pl, probadist)
        A = probadist.atoms
        As = sort(A, by=get_angle)
        p = probadist.weights
        plot!(
            pl,
            vcat(map(first, As), first(As[1])),
            vcat(map(last, As), last(As[1]));
            fillrange=0,
            fillcolor=:blue,
            fillalpha=0.1,
            linestyle=:dash,
            linecolor=logocolors.blue,
            label=L"\mathrm{conv}(\hat{p}(\theta))"
        )
        scatter!(
            pl,
            map(first, A),
            map(last, A);
            markersize=25 .* p .^ 0.5,
            markercolor=logocolors.blue,
            markerstrokewidth=0,
            markeralpha=0.4,
            label=L"\hat{p}(\theta)"
        )
    end

    function plot_expectation!(pl, probadist)
        ŷΩ = compute_expectation(probadist)
        scatter!(
            pl,
            [ŷΩ[1]],
            [ŷΩ[2]];
            color=logocolors.blue,
            markersize=6,
            markershape=:hexagon,
            label=L"\hat{f}(\theta)"
        )
    end

    function compress_distribution!(
        probadist::FixedAtomsProbabilityDistribution{A,W}; atol=0
    ) where {A,W}
        (; atoms, weights) = probadist
        to_delete = Int[]
        for i in length(probadist):-1:1
            ai = atoms[i]
            for j in 1:(i-1)
                aj = atoms[j]
                if isapprox(ai, aj; atol=atol)
                    weights[j] += weights[i]
                    push!(to_delete, i)
                    break
                end
            end
        end
        sort!(to_delete)
        deleteat!(atoms, to_delete)
        deleteat!(weights, to_delete)
        return probadist
    end

	set_angle_oracle = md"""
angle = $(@bind angle_oracle Slider(0:0.01:2π; default=π, show_value=false))
""";
	set_angle_perturbed = md"""
angle = $(@bind angle_perturbed Slider(0:0.01:2π; default=π, show_value=false))
""";
	set_nb_samples_perturbed = md"""
samples = $(@bind nb_samples_perturbed Slider(1:500; default=10, show_value=true))
""";
	set_epsilon_perturbed = md"""
epsilon = $(@bind epsilon_perturbed Slider(0.0:0.02:1.0; default=0.0, show_value=true))
""";
	set_plot_probadist_perturbed = md"""
Plot probability distribution? $(@bind plot_probadist_perturbed CheckBox())
""";
end;

# ╔═╡ 02e3fb9f-943b-4308-b7ba-5651afc62c31
ChooseDisplayMode()

# ╔═╡ 3b6b897d-8c62-4589-b8a7-3fa9ccea40a0
md"""
- Each green question box expects a written answer. For this, replace the `still_missing()` yellow box after by `md"Your answer"`.
- TODO boxes expect some code implementation, and eventually some comments and analyis.
"""

# ╔═╡ e5ba2253-448b-42a7-b626-80cbf9eecaa2
tip(md"""This file is a [Pluto](https://plutojl.org/) notebook. There are some differences respect to Jupyter notebooks you may be familiar with:
- It's a regular julia code file.
- **Self-contained** environment: packages are managed automatically.
- **Reactivity** and interactivity: cells are connected, such that when you modify a variable value, all other cells depending on it (i.e. using this variable) are automatically reloaded and their outputs updated. Feel free to modify some variables to observe the effects on the other cells. This allow interactivity with tools such as dropdown and sliders.
See the [Pluto documentation for more details](https://plutojl.org/en/docs/).
""")

# ╔═╡ 8ab3b55e-b771-4da2-8fbe-b7f7823c8256
md"""
# 1. Recap on CO-ML pipelines
"""

# ╔═╡ 5ea673bf-861a-4d94-9c68-727fbfeb2a47
md"""

**Points of view**: 
1. Enrich learning pipelines with combinatorial algorithms.
2. Enhance combinatorial algorithms with learning pipelines.

```math
\xrightarrow[x]{\text{Instance}}
\fbox{ML predictor}
\xrightarrow[\theta]{\text{Objective}}
\fbox{CO algorithm}
\xrightarrow[y]{\text{Solution}}
```

**Challenge:** Differentiating through CO algorithms.

**Two main learning settings:**
- Learning by imitation: instances with labeled solutions $(x_i, y_i)_i$.
- Learning by experience: no labeled solutions $(x_i)_i$.
"""

# ╔═╡ 637ec3d4-b4dc-42da-aa26-ab2a85b75150
md"""
## Many possible applications in both fields

- Shortest paths on Warcraft maps
- Stochastic Vehicle Scheduling
- Two-stage Minimum Spanning Tree
- Single-machine scheduling
- Dynamic Vehicle Routing
- ...
"""

# ╔═╡ e1db44f8-7426-4e8e-9d99-415d91377645
md"""
## Smoothing by regularization

```math
\xrightarrow[\text{instance $x$}]{\text{Problem}}
\fbox{NN $\varphi_w$}
\xrightarrow[\text{direction $\theta$}]{\text{Objective}}
\fbox{MILP $\underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y$}
\xrightarrow[\text{solution $\widehat{y}$}]{\text{Candidate}}
```

The combinatorial layer function

```math
f\colon \theta\longmapsto \underset{y \in \mathcal{Y}}{\mathrm{argmax}} ~ \theta^\top y
```
is piecewise constant $\implies$ no gradient information.

The perturbed regularized optimizer is defined by:

```math
\hat{f}_\varepsilon(\theta) = \mathbb{E}_{Z}\big[ \underset{y \in \mathcal{Y}}{\mathrm{argmax}} (\theta + \varepsilon Z)^\top y \big]
```
with ``Z\sim\mathcal{N}(0, 1)``, ``\varepsilon>0``.

``\implies`` becomes differentiable.

Can be seen as an expectation over the vertices of $\mathrm{conv}(\mathcal{Y})$.

```math

\hat{f}_\varepsilon(\theta) = \mathbb{E}_{\hat{p}(\cdot|\theta)}[Y] = \sum_{y\in\mathcal{Y}}~y~\hat{p}(y|\theta)
```
"""

# ╔═╡ 07fdfdff-8860-46ec-bca5-6ebc5e355bcc
md"""
## Linear oracle
"""

# ╔═╡ 90fb8785-e3d3-4e40-bbf7-aba19221944c
md"""Let's build a polytope with `N` vertices, and visualize perturbations and loss over it."""

# ╔═╡ a70ac2a7-33fc-423b-a6a7-c2093b2d48f8
N = 7

# ╔═╡ 8b5732a7-7b9a-486a-b5d3-5587858f9320
polytope = [[cospi(2k / N), sinpi(2k / N)] for k in 0:N-1];

# ╔═╡ 69a6df2d-8f46-4859-bd5d-a63512bde6bc
md"""Combinatorial oracle: ``f(\theta; x) = \arg\max_{y\in\mathcal{Y}(x)} \theta^\top y``"""

# ╔═╡ d81abd36-764b-469d-81d6-e67e4cf81b7a
maximizer(θ; polytope) = polytope[argmax(dot(θ, v) for v in polytope)];

# ╔═╡ aedefc27-2e69-4b3f-9f3f-51d2d551dcbb
md"""
Here is a figure of the polytope and the armax output of the oracle in red.

You can modify θ by using the slider below to modify its angle:
"""

# ╔═╡ c8481064-56c2-49ab-bde2-e090db6c6762
let
	θ = 0.5 .* [cos(angle_oracle), sin(angle_oracle)]
	pl = init_plot("Linear oracle")
	plot_polytope!(pl, polytope)
	plot_objective!(pl, θ)
	plot_maximizer!(pl, θ, polytope, maximizer)
	pl
end

# ╔═╡ 4c019cc6-a534-4d13-adfa-d1ac9dd5bf89
set_angle_oracle

# ╔═╡ 4d000eb7-c88a-4126-a43e-2dff91e961b6
md"""We use the [`Zygote.jl`](https://fluxml.ai/Zygote.jl/stable/) automatic differentiation library to compute the jacobian of our CO oracle with respect to ``\theta``.
"""

# ╔═╡ 5c57f15d-80b9-4b20-834b-50a575cde401
let
	θ = 0.5 .* [cos(angle_oracle), sin(angle_oracle)]
	jac = Zygote.jacobian(θ -> maximizer(θ; polytope), θ)[1]
	@info "" θ=θ jacobian=jac
end

# ╔═╡ 27f01e89-7af6-4c48-b4a3-d48778e53fd1
question_box(md"1) Why is the jacobian zero for all values of ``\theta``?")

# ╔═╡ 5a9f0606-6d36-4e01-a0e5-c504beb1ee16
md"""## Perturbed Layer"""

# ╔═╡ 15ed7861-cdd6-45d5-b3bc-d8ebb21638fa
md"""[`InferOpt.jl`](https://github.com/axelparmentier/InferOpt.jl) provides the `PerturbedAdditive` wrapper to regularize any given combinatorial optimization oracle $f$, and transform it into $\hat f$.

It takes the maximizer as the main arguments, and several optional keyword arguments such as:
- `ε`: size of the perturbation (=1 by default)
- `nb_samples`: number of Monte Carlo samples to draw for estimating expectations (=1 by default)

See the [documentation](https://axelparmentier.github.io/InferOpt.jl/dev/) for more details.
"""

# ╔═╡ 3958ad45-3774-4f04-ab93-572b9abc2fce
perturbed_layer = PerturbedAdditive(
	maximizer;
	ε=epsilon_perturbed,
	nb_samples=nb_samples_perturbed,
	seed=0
)

# ╔═╡ 9e34e85c-e426-4b41-a014-93e9bbc892a6
md"""Now we can visualize the perturbed maximizer output"""

# ╔═╡ d7cbabfb-1fe6-4118-af40-b68358cf3c31
TwoColumn(set_angle_perturbed, set_epsilon_perturbed)

# ╔═╡ 1e3c4143-d948-4df4-9c25-c15838d55a1a
TwoColumn(set_nb_samples_perturbed, set_plot_probadist_perturbed)

# ╔═╡ 2460ccdd-b6eb-4f33-9d8c-a3b4db2cfd18
let
	θ = 0.5 .* [cos(angle_perturbed), sin(angle_perturbed)]
	probadist = compute_probability_distribution(
		perturbed_layer, θ; polytope,
	)
	compress_distribution!(probadist)
	pl = init_plot("Perturbation")
	plot_polytope!(pl, polytope)
	plot_objective!(pl, θ)
	plot_probadist_perturbed && plot_distribution!(pl, probadist)
	plot_maximizer!(pl, θ, polytope, maximizer)
	plot_expectation!(pl, probadist)
	pl
end

# ╔═╡ b52ca8d8-a379-4aae-ae8f-6678871663cb
md"""The perturbed maximizer is differentiable:"""

# ╔═╡ 8924b1fe-64fb-43f5-989a-0503edf7f753
let
	θ = 0.5 .* [cos(angle_perturbed), sin(angle_perturbed)]
	Zygote.jacobian(θ -> perturbed_layer(θ; polytope), θ)[1]
end

# ╔═╡ 27a50144-2933-4bdd-a5b3-48c0d0d4dc7d
question_box(md"2. What can you say about the derivatives of the perturbed maximizer?")

# ╔═╡ 967c92c9-8ea1-4ec4-a524-3065d60186f0
md"""
## Fenchel-Young loss (learning by imitation)
By defining:

```math
F^+_\varepsilon (\theta) := \mathbb{E}_{Z}\big[ \operatorname{max}_{y \in \mathcal{Y}(x)} (\theta + \varepsilon Z)^\top y \big],
```
and ``\Omega_\varepsilon^+`` its Fenchel conjugate, we can define the Fenchel-Young loss as follows:
```math
\mathcal{L}_{\varepsilon}^{\text{FY}}(\theta, \bar{y}) = F^+_\varepsilon (\theta) + \Omega_\varepsilon(\bar{y}) - \theta^\top \bar{y}
```

Given a target solution $\bar{y}$ and a parameter $\theta$, a subgradient is given by:
```math
\widehat{f}(\theta) - \bar{y} \in \partial_\theta \mathcal{L}_{\varepsilon}^{\text{FY}}(\theta, \bar{y}).
```
The optimization block has meaningful gradients $\implies$ we can backpropagate through the whole pipeline, using automatic differentiation.
"""

# ╔═╡ 94a177ca-2c03-45a6-830f-e59e49cbb2fa
question_box(md"3. What are the properties of ``\mathcal{L}_{\varepsilon}^{\text{FY}}?``")

# ╔═╡ fef7491c-fecc-4812-93eb-d9271cfdf7b8
md"""Let's define the Fenchel-Young loss by using the `FenchelYoungLoss` wrapper from `InferOpt`:"""

# ╔═╡ 6344f751-4c36-4cf3-b700-ad28e102a91f
fyl = FenchelYoungLoss(perturbed_layer)

# ╔═╡ 49a80eac-3fdc-49c9-87e7-1635a75cfb13
md"""Let's visualize a contour plot of the loss with target ȳ fixed."""

# ╔═╡ 420c9aff-bd54-4378-b12a-3088f39dc52e
X, Y = range(-1, 1, length=100), range(-1, 1, length=100);

# ╔═╡ b5f72c4c-a7f1-48dc-bd54-a41d52136732
TwoColumn(md"""Change `y_index` value to change target vertex ȳ:""", md"y\_index = $(@bind y_index Select(1:N))")

# ╔═╡ 876a600d-b52e-431a-882e-739edb4546c7
f(x, y) = fyl([x, y], polytope[y_index]; polytope);

# ╔═╡ 8a0c8e33-f4b9-480c-8711-6789e870c3fe
Z = @. f(X', Y);

# ╔═╡ 640779ef-35a8-4609-ad88-09a320b29c04
TwoColumn(set_nb_samples_perturbed, set_epsilon_perturbed)

# ╔═╡ a4f8fcca-a424-45af-96fa-b285f7f24202
contour(X, Y, Z; color=:turbo, fill=true, xlabel="θ₁", ylabel="θ₂")

# ╔═╡ ab50793b-b4c6-4a42-9878-6b58e00077fd
question_box(md"4. What happens when $\varepsilon = 0$? What happens when $\varepsilon$ increases?")

# ╔═╡ f23f322a-6f9e-4cac-8fa1-cceea8f7a3ab
md"# 2. Path-finding on Warcraft maps"

# ╔═╡ d8f21cc2-dc65-450b-b502-0cc47b6b9d21
b = WarcraftBenchmark()

# ╔═╡ ba9fec97-c64f-4a97-b39f-6bdfc7b078b1
dataset_size = 100

# ╔═╡ 2b6a9353-98e2-44fe-9ffb-f226275c3e91
dataset = generate_dataset(b, dataset_size)

# ╔═╡ 8ef00799-f836-4206-8dd4-04f2c910a749
typeof(dataset)

# ╔═╡ c3a27c83-fb98-47c0-a1c8-1d73fc736ab2
md"index = $(@bind index Slider(1:100; show_value=true))"

# ╔═╡ 95386f17-d202-4a22-906f-c894871ac565
plot_data(b, dataset[index])

# ╔═╡ 4f81bf09-f00d-4a63-bee6-f2b2483ac40e
model = generate_statistical_model(b)

# ╔═╡ 47249d4d-3c52-4be1-8875-71172b1d9b91
dijkstra_maximizer = generate_maximizer(b; dijkstra=true)

# ╔═╡ f989e5d3-6d64-45e6-a030-049e5ad2e9d9
bellman_maximizer = generate_maximizer(b; dijkstra=false)

# ╔═╡ bb39af0c-697c-4ac7-8de1-259f564322e9
θ = dataset[index].θ

# ╔═╡ 2d4f2f0c-a396-421b-920e-8cb34e85a05e
dijkstra_maximizer(θ)

# ╔═╡ dc117fac-fa29-4848-a94c-e490caa52814
bellman_maximizer(θ)

# ╔═╡ Cell order:
# ╠═87660d1a-acca-11ef-02da-d113baade226
# ╠═5a34ecaa-55ca-4c0e-b4eb-59bae1b1c926
# ╠═edc13965-3c29-4ddd-ad7e-d1f5b4b9e6c9
# ╠═481c897f-0920-4c43-854c-49af36405cab
# ╟─32eba97b-e192-429f-86a6-816545c75761
# ╟─02e3fb9f-943b-4308-b7ba-5651afc62c31
# ╟─3b6b897d-8c62-4589-b8a7-3fa9ccea40a0
# ╟─e5ba2253-448b-42a7-b626-80cbf9eecaa2
# ╟─8ab3b55e-b771-4da2-8fbe-b7f7823c8256
# ╟─5ea673bf-861a-4d94-9c68-727fbfeb2a47
# ╟─637ec3d4-b4dc-42da-aa26-ab2a85b75150
# ╟─e1db44f8-7426-4e8e-9d99-415d91377645
# ╟─07fdfdff-8860-46ec-bca5-6ebc5e355bcc
# ╟─90fb8785-e3d3-4e40-bbf7-aba19221944c
# ╠═a70ac2a7-33fc-423b-a6a7-c2093b2d48f8
# ╠═8b5732a7-7b9a-486a-b5d3-5587858f9320
# ╟─69a6df2d-8f46-4859-bd5d-a63512bde6bc
# ╠═d81abd36-764b-469d-81d6-e67e4cf81b7a
# ╟─aedefc27-2e69-4b3f-9f3f-51d2d551dcbb
# ╟─c8481064-56c2-49ab-bde2-e090db6c6762
# ╟─4c019cc6-a534-4d13-adfa-d1ac9dd5bf89
# ╟─4d000eb7-c88a-4126-a43e-2dff91e961b6
# ╠═5c57f15d-80b9-4b20-834b-50a575cde401
# ╟─27f01e89-7af6-4c48-b4a3-d48778e53fd1
# ╟─5a9f0606-6d36-4e01-a0e5-c504beb1ee16
# ╟─15ed7861-cdd6-45d5-b3bc-d8ebb21638fa
# ╠═3958ad45-3774-4f04-ab93-572b9abc2fce
# ╟─9e34e85c-e426-4b41-a014-93e9bbc892a6
# ╟─d7cbabfb-1fe6-4118-af40-b68358cf3c31
# ╟─1e3c4143-d948-4df4-9c25-c15838d55a1a
# ╟─2460ccdd-b6eb-4f33-9d8c-a3b4db2cfd18
# ╟─b52ca8d8-a379-4aae-ae8f-6678871663cb
# ╠═8924b1fe-64fb-43f5-989a-0503edf7f753
# ╟─27a50144-2933-4bdd-a5b3-48c0d0d4dc7d
# ╟─967c92c9-8ea1-4ec4-a524-3065d60186f0
# ╟─94a177ca-2c03-45a6-830f-e59e49cbb2fa
# ╟─fef7491c-fecc-4812-93eb-d9271cfdf7b8
# ╠═6344f751-4c36-4cf3-b700-ad28e102a91f
# ╟─49a80eac-3fdc-49c9-87e7-1635a75cfb13
# ╠═420c9aff-bd54-4378-b12a-3088f39dc52e
# ╟─b5f72c4c-a7f1-48dc-bd54-a41d52136732
# ╠═876a600d-b52e-431a-882e-739edb4546c7
# ╠═8a0c8e33-f4b9-480c-8711-6789e870c3fe
# ╠═640779ef-35a8-4609-ad88-09a320b29c04
# ╠═a4f8fcca-a424-45af-96fa-b285f7f24202
# ╟─ab50793b-b4c6-4a42-9878-6b58e00077fd
# ╟─f23f322a-6f9e-4cac-8fa1-cceea8f7a3ab
# ╠═f608b9d7-2266-4686-8241-d0aa14501103
# ╠═d8f21cc2-dc65-450b-b502-0cc47b6b9d21
# ╠═ba9fec97-c64f-4a97-b39f-6bdfc7b078b1
# ╠═2b6a9353-98e2-44fe-9ffb-f226275c3e91
# ╠═8ef00799-f836-4206-8dd4-04f2c910a749
# ╠═95386f17-d202-4a22-906f-c894871ac565
# ╠═4f81bf09-f00d-4a63-bee6-f2b2483ac40e
# ╠═47249d4d-3c52-4be1-8875-71172b1d9b91
# ╠═f989e5d3-6d64-45e6-a030-049e5ad2e9d9
# ╠═bb39af0c-697c-4ac7-8de1-259f564322e9
# ╠═2d4f2f0c-a396-421b-920e-8cb34e85a05e
# ╟─c3a27c83-fb98-47c0-a1c8-1d73fc736ab2
# ╠═dc117fac-fa29-4848-a94c-e490caa52814
