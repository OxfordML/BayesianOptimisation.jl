module t
import BayesianOptimisation; const bo = BayesianOptimisation
import GaussianProcess; const gp = GaussianProcess
import Optim
using PyPlot
const obs_noise = 0

function observer(model, x::Array{Float64, 1}, y::Float64)
	observer(model, x, [y])
end
function observer(model, x::Array{Float64, 1}, y::Array{Float64, 1})
    n = size(y, 1)
    X = repmat(x', n)
    gp.observe!(model, X, y)
end

function inferer(model)
    gp.infer!(gp.MLE2([-10, -10, -10], [10, 10, 10]), model)
end

function predictor(model, x::Array{Float64,}, grad)
    d = length(x)
    Xp = reshape(x, (1, d))	
    ret = gp.predict(model, Xp)
    ret[1][1], max(ret[2][1], eps())
end

function objective(x::Array{Float64, }, noise=obs_noise)
    -0.7exp(-x[1].^2) +
    -0.6exp(-(x[1] - 4).^2) +
    -0.99exp(-(x[1] - 2).^2) +
    -0.5exp(-(x[1] + 4).^2) +
    -1.1exp(-(x[1] + 2).^2)+
    randn() * noise
end

plotspace = linspace(-5, 5, 100)''

pre_samp = 2
n_rounds = 15
reps = 1

u = linspace(-5, 5, 30)''
kernel = gp.MaternIso32(0.277, 0.277)
lik = gp.GaussianLikelihood(-7.28)

nmodels = 4
traces = Array(bo.BOTrace, nmodels, n_rounds, reps)
for rep = 1:reps
	 models = [gp.FullModel(kernel, lik), 
			gp.NystromModel(kernel, lik, [5.], [-5.], neig_samples=30),
			gp.LaplacianModel(kernel, lik, gp.OneD(8), 30),
			gp.FITCModel(kernel, lik, u)]
	X = (rand(pre_samp, 1) .- 0.5) * 10
	y = Float64[objective(X[r, :]) for r = 1:pre_samp]
	for (i,model) in enumerate(models)
            gp.observe!(model, X, y)
            gp.infer!(gp.MLE2([-10, -10, -10], [10, 10, 10]), model)
            function callback(o::bo.BO, round, step_time)
                    figure(1);clf()
                    predictions = [predictor(model, plotspace[r, :], []) for r=1:100]
                    mean = Float64[predictions[i][1] for i = 1:100]
                    var = Float64[predictions[i][2] for i = 1:100]
                    f = Float64[objective(plotspace[r, :]) for r=1:100]
                    stddev = sqrt(var)
                    plot(plotspace, f, "g")
                    plot(plotspace, mean, "k")
                    fill_between(plotspace[:], (mean + 2stddev)[:], (mean - 2stddev)[:], alpha=0.5)
                    plot(plotspace, mean + 2stddev, "k--")
                    plot(plotspace, mean - 2stddev, "k--")
                    println("$round, $(typeof(model))")
            end
            bayes_opt = bo.DeterministicBO(bo.EL(), 5.0, [5.], [-5.], callback)
            traces[i, :, rep] = bo.optimize!(bayes_opt, n_rounds, objective, 
                    (args...)->predictor(model, args...), 
                    (args...)->observer(model, args...),
                    (args...)->inferer(model, args...))
	end
end

end

