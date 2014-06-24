reload("../src/BayesianOptimisation.jl")
module t
include("../../PaperCode/util/oned_opt_func.jl")
import BayesianOptimisation; const bo = BayesianOptimisation
import GaussianProcess; const gp = GaussianProcess

const obs_noise = 0.1

kernel = gp.MaternIso32(-2, -2)
lik = gp.GaussianLikelihood(-2.9)
c = gp.Cartesian(1, 13)
model = gp.LaplacianModel(kernel, lik, c, 200)
function observer(xin::Array{Float64, }, yin::Float64, learn=true)
	d = length(xin)
	X = reshape(xin, (1, d))
	Y = [yin]
	gp.observe!(model, X, Y)
	if learn
		gp.infer!(gp.MLE2(), model)
	end
end

function predictor(x::Array{Float64,}, grad)
	d = length(x)
	Xp = reshape(x, (1, d))
	ret = gp.predict(model, Xp)
	ret[1][1], ret[2][1]
end

function objective(x::Array{Float64, }, noise=obs_noise)
	(opt_me(x)[1] .+ randn() * noise)::Float64
end


bayes_opt = bo.BO(bo.EL(), 100.0, [10.], [-10.])

# bo.optimize!(bayes_opt, 1, objective, predictor, observer)
bo.optimize!(bayes_opt, 5000, objective, predictor, observer)
end

