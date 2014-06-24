module BayesianOptimisation
import Distributions, NLopt
# const plt = PyPlot
export EL, UCB, LCB, BO, optimize!
abstract Utility
type EL <: Utility end
type UCB <: Utility
	ratio
end
type LCB <: Utility
	ratio
end


type BestValue
	value::Float64
	loc::Array{Float64,}
end
import Base.copy
copy(b::BestValue) = BestValue(copy(b.value), copy(b.loc))
abstract BO
type DeterministicBO <: BO
	utility::Utility
	d::Integer
	lowest_mean::BestValue
	best_next::BestValue
	best_var::BestValue
	variance_limit::Float64
	ub::Array{Float64, 1}
	lb::Array{Float64, 1}
	cb::Function
	lowest_seen::FloatingPoint
	meandirect::Symbol
end

type StochasticBO <: BO
	nevals::Integer
	method::Symbol
	utility::Utility
	d::Integer
	lowest_mean::BestValue
	best_next::BestValue
	best_var::BestValue
	variance_limit::Float64
	ub::Array{Float64, 1}
	lb::Array{Float64, 1}
	cb::Function
	lowest_seen::FloatingPoint
	meandirect::Symbol
end

immutable BOTrace
	step::Integer
	lowest_mean::BestValue
	step_time::FloatingPoint
	lowest_seen::FloatingPoint
end

function DeterministicBO(u::Utility, v_limit::Number, ub::Array{Float64, 1}, 
				lb::Array{Float64, 1}, cb=(args...)->None, meandirect=:GN_DIRECT)
	d = length(ub)
	@assert length(ub) == length(lb) "Upper bounds, lower bounds must be same size"
	DeterministicBO(u, d, BestValue(Inf, fill(0, (d,))), 
				BestValue(Inf, fill(0, (d,))), BestValue(Inf, fill(0, (d,))), v_limit, ub, lb, cb, Inf, meandirect)
end

function StochasticBO(nevals::Integer, method::Symbol, u::Utility, 
	v_limit::Number, 
	ub::Array{Float64, 1}, lb::Array{Float64, 1}, cb=(x)->None, meandirect=:GN_DIRECT)
	d = length(ub)
	@assert length(ub) == length(lb) "Upper bounds, lower bounds must be same size"
	StochasticBO(nevals, method, u, d, BestValue(Inf, fill(0, (d,))), 
				BestValue(Inf, fill(0, (d,))), BestValue(Inf, fill(0, (d,))), v_limit, ub, lb, cb, Inf, meandirect)
end

function optimize!(o::BO, rounds, objective::Function, 
					predictor::Function, observer::Function, 
					inferer::Function, run_time=Inf)
	start_time = time()
	trace = Array(BOTrace, (rounds, ))
	for round = 1:rounds
		step_time = @elapsed begin
			find_best!(o, predictor)
			next_move!(o, predictor)
			observe!(o, objective, observer)
			inferer()
		end
		o.cb(o, round, step_time)
		if time() - start_time > run_time
			break
		end
		trace[round] = BOTrace(round, copy(o.lowest_mean), step_time, o.lowest_seen)
	end
	trace
end

# function variance_cap(o, predictor, x)
# 	pred = predictor(x, [])
# 	if pred[2] < o.variance_limit
# 		return pred[1]
# 	else
# 		return Inf
# 	end
# end
# Finds the lowest point on the regressor
function find_best!(o::BO, predictor::Function)
	opt = NLopt.Opt(o.meandirect, o.d)
	NLopt.min_objective!(opt, (x, grad)->predictor(x, [])[1])
	NLopt.upper_bounds!(opt, o.ub)
	NLopt.lower_bounds!(opt, o.lb)
	mid = (o.ub + o.lb) ./ 2
	NLopt.ftol_rel!(opt, 1e-15)
	NLopt.maxtime!(opt, 20)
	if o.meandirect == :GN_ORIG_DIRECT
		NLopt.inequality_constraint!(opt,
			(x, grad)->predictor(x, grad)[2] - o.variance_limit, 
			1e-8)
	end
	try
		res = NLopt.optimize(opt,mid)
		mean = res[1]
		o.lowest_mean.value = res[1]
		if mean != Inf
			if o.meandirect == :GN_ORIG_DIRECT
				o.lowest_mean.loc[:] = rescale01d(o, res[2][:])
			else
				o.lowest_mean.loc[:] = res[2][:]
			end
		else
			println("Variance threshold too low")
			o.lowest_mean.loc[:] = res[2][:]
			var = predictor(res[2][:], [])[2]
			println("Variance: $var")
		end
		println("Mean opt finish at $(res[3]) value $(res[1])")
	catch e
		println(e)
		if isa(e, InterruptException)
			throw(e)
		end
	end
end
function next_move!(o::BO, predictor::Function)
	try
		if o.lowest_mean.value == Inf
			next_move_var!(o, predictor)
		else
			next_move_utility!(o, predictor)
		end
	catch e
	end

end
function next_move_utility!(o::BO, predictor::Function)
	opt = NLopt.Opt(:GN_DIRECT, o.d)
	f(x, grad) = utility(o.utility, predictor(x, grad)..., o.lowest_mean.
			value)
	NLopt.min_objective!(opt, f)
	NLopt.upper_bounds!(opt, o.ub)
	NLopt.lower_bounds!(opt, o.lb)
	mid = (o.ub + o.lb) ./ 2
	NLopt.ftol_rel!(opt, 1e-15)
	NLopt.maxtime!(opt, 20)
	res = NLopt.optimize(opt, mid)
	println("Utility opt finish with $(res[3]), mean $(predictor(res[2], [])[1])")
	o.best_next.loc[:] = res[2][:]
	o.best_next.value = res[1]
end

# function next_move_var!(o::BO, predictor::Function)
# 	opt = NLopt.Opt(:GN_DIRECT, o.d)
# 	f(x, grad) = predictor(x, grad)[2]
# 	NLopt.max_objective!(opt, f)
# 	NLopt.upper_bounds!(opt, o.ub)
# 	NLopt.lower_bounds!(opt, o.lb)
# 	NLopt.ftol_rel!(opt, 1e-14)
# 	NLopt.maxtime!(opt, 10)
# 	res = NLopt.optimize(opt, random_location(o))
# 	o.best_next.loc[:] = res[2][:]
# 	o.best_next.value = res[1]
# end

function observe!(o::StochasticBO, objective, observer)
	obs = zeros(o.nevals,)
	for i = 1:o.nevals
		obs[i] = objective(o.best_next.loc[:])
	end
	o.lowest_seen = min(obs..., o.lowest_seen)
	if o.method == :AVG
		observer(o.best_next.loc, mean(obs))
	elseif o.method == :FULL
		observer(o.best_next.loc, obs)
	end
end
function observe!(o::DeterministicBO, objective, observer)
	# obs = zeros(1000,)
	# for i = 1:1000
	# 	obs[i] = objective(o.best_next.loc[:])
	# end
	obs = objective(o.best_next.loc[:])
	o.lowest_seen = min(obs, o.lowest_seen)
	observer(o.best_next.loc, obs)
end
function random_location(o)
	rescale01d(o, rand(o.d, ))
end

function rescale01d(o, loc)
	scale = o.ub - o.lb
	offset = (o.ub + o.lb) / 2
	((loc .- 0.5) .* scale) .+ offset
end

function utility(::EL, mean, variance, best)
	n = Distributions.Normal(mean, sqrt(variance))
	best .+ (mean-best) * Distributions.cdf(n, best) .- variance * 
		Distributions.pdf(n, best)
end

function utility(u::UCB, mean, variance, best)
	mean + u.ratio * sqrt(variance)
end

function utility(u::LCB, mean, variance, best)
	mean - u.ratio * sqrt(variance)
end

end