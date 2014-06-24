reload("../src/BayesianOptimisation.jl")
module t
include("../../PaperCode/util/oned_opt_func.jl")
import BayesianOptimisation; const bo = BayesianOptimisation
import GaussianProcess; const gp = GaussianProcess
using PyPlot
const obs_noise = 1.0


function observer(xin::Array{Float64, }, yin::Float64, learn=true)
	d = length(xin)
	X = reshape(xin, (1, d))
	Y = [yin]
	gp.observe!(model, X, Y)
	if learn
		# gp.infer!(gp.MLE2(), model)
	end
end

function predictor(x::Array{Float64,}, grad)
	d = length(x)
	Xp = reshape(x, (1, d))
	ret = gp.predict(model, Xp)
	ret[1][1], ret[2][1]
end

function objective(x::Array{Float64, }, noise=obs_noise)
	-exp(-x[1].^2) + 0.99 * (
	-exp(-(x[1] - 5).^2) +
	-exp(-(x[1] - 2.5).^2) +
	-exp(-(x[1] + 5).^2) +
	-exp(-(x[1] + 2.5).^2) )+
	 randn() * noise
end

X = (rand(1000, 1) -0.5) * 20
y = Float64[objective(X[r, :]) for r=1:size(X,1)]

noise = var(y)
kernel = gp.MaternIso32(0, 0)
lik = gp.GaussianLikelihood(log(sqrt(obs_noise)))
model = gp.NystromModel(kernel, lik, 10, neig_samples=)
# model = gp.FullModel(kernel, lik)

gp.observe!(model, X, y)
bayes_opt = bo.BO(bo.EL(), 100.0, [10.], [-10.])

# bo.optimize!(bayes_opt, 1, objective, predictor, observer)
bo.optimize!(bayes_opt, 1000, objective, predictor, observer, 1)

kernel = gp.MaternIso32(log(4), 0)
lik = gp.GaussianLikelihood(log(sqrt(obs_noise)))
# model = gp.NystromModel(kernel, lik, 10, neig_samples=50)
# model = gp.FullModel(kernel, lik)

# gp.observe!(model, X, y)
# bayes_opt = bo.BO(bo.EL(), 100.0, [10.], [-10.])
# # bo.optimize!(bayes_opt, 1, objective, predictor, observer)
# bo.optimize!(bayes_opt, 1, objective, predictor, observer, 2)
end

