import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import os
import tabular_mdp
from jax import jacfwd, jacrev


"""
Utility functions for numeric computations
"""
def _safe_ratio(pi, mu):
	return pi / (mu + 1e-8)


def _normalize(ratios, pi, mu):
	ratios = np.array(ratios)
	all_ratios = _safe_ratio(pi, mu)
	ratios /= np.sum(all_ratios, axis=-1)
	return list(ratios)


"""
Low-level utility functions for computing estimates as differentiable parameters
"""
def Vtrace_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on V-trace (Espeholt et al, 2018).

	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average Vtrace value estimates at the initial state of the trajectories
	"""
	evaluations = []
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = jnp.min(jnp.array([ratio, rho]))
			c_bar = jnp.min(jnp.array([ratio, c]))
			v_estimate = V[s] + rho_bar * (r + gamma * V[s_next]- V[s]) + gamma * c_bar * (v_estimate - V[s_next])
		evaluations.append(v_estimate)
	return jnp.mean(jnp.array(evaluations))


def Firstorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on first-order Taylor expansion of value function  (Kakade et al, 2002; Tang et al, 2020).
	First-order Taylor expansion is commonly used in policy optimization algorithms such as TRPO and PPO.

	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average first-order vaue estimates at the initial state of the trajectories
	"""
	evaluations = []
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[states[-1]]
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = jnp.min(jnp.array([ratio, rho]))
			c_bar = jnp.min(jnp.array([ratio, c]))
			v_estimate = V[s] + 1.0 * (r + gamma * V[s_next]- V[s]) + gamma * 1.0 * (v_estimate - V[s_next])
			new_estimate = jax.lax.stop_gradient(v_estimate - V[s]) * (ratio - 1.0)
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			init_estimate += gamma**step * estimate
		evaluations.append(init_estimate)
	return jnp.mean(jnp.array(evaluations))


def Firstorder_evaluation_util(pi, mu, T, gamma, V, trajs, rho, c, stop_grad):
	"""
	Utility function used by Secondorder_evaluation
	"""
	evaluations = []
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[-1]
		time = -1
		all_evaluations = [v_estimate]
		evaluations_now = 0.0
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			time -= 1
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = jnp.min(jnp.array([ratio, rho]))
			c_bar = jnp.min(jnp.array([ratio, c]))
			v_estimate = V[time] + 1.0 * (r + gamma * V[time+1]- V[time]) + gamma * 1.0 * (v_estimate - V[time+1])
			if stop_grad:
				multiplier = jax.lax.stop_gradient(v_estimate - V[time])
			else:
				multiplier = v_estimate - V[time]
			new_estimate = multiplier * (ratio - 1.0)
			evaluations_now = gamma * evaluations_now + new_estimate
			all_evaluations.append(evaluations_now + V[time])
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		all_evaluations = all_evaluations[::-1]
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			init_estimate += gamma**step * estimate
		evaluations.append(init_estimate)
	return jnp.mean(jnp.array(evaluations)), all_evaluations


def Secondorder_util(pi, mu, T, gamma, V, Vbase, trajs, rho, c, stop_grad):
	"""
	Utility function used by Secondorder_evaluation
	"""
	evaluations = []
	num_simulations = len(trajs)
	for i in range(num_simulations):
		traj = trajs[i]
		states, actions, rewards = traj['states'], traj['actions'], traj['rewards']
		all_estimates = []
		v_estimate = V[-1]
		time = -1
		all_evaluations = [v_estimate]
		evaluations_now = 0.0
		for s,a,r,s_next in zip(states[:-1][::-1], actions[::-1], rewards[::-1], states[1:][::-1]):
			time -= 1
			ratio = _safe_ratio(pi[s, a], mu[s, a])
			rho_bar = jnp.min(jnp.array([ratio, rho]))
			c_bar = jnp.min(jnp.array([ratio, c]))
			v_estimate = r + gamma * V[time+1]
			if stop_grad:
				multiplier = jax.lax.stop_gradient(v_estimate - Vbase[time])
			else:
				multiplier = v_estimate - Vbase[time]
			new_estimate = multiplier * (ratio - 1.0)
			evaluations_now = gamma * evaluations_now + new_estimate
			all_evaluations.append(evaluations_now + V[time])
			all_estimates.append(new_estimate)
		all_estimates = all_estimates[::-1]
		all_evaluations = all_evaluations[::-1]
		init_estimate = 0.0
		for step,estimate in enumerate(all_estimates):
			init_estimate += gamma**step * estimate
		evaluations.append(init_estimate)
	return jnp.mean(jnp.array(evaluations)), all_evaluations


def Secondorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c):
	"""
	This evaluation subroutine is based on second-order Taylor expansion of value function  (Tang et al, 2020).

	Args:
		pi: target policy
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		Average second-order estimates at the initial state of the trajectories
	"""
	# Here, we adopt a recursive implementation of second-order expansion
	# we first evaluate the first-order values
	_, all_evals = Firstorder_evaluation_util(pi, mu, T, gamma, V, trajs, rho, c, stop_grad=True)
	# all_evals is a list of value functions along the trajectory, computed based on first-order
	# we input all_evals back into the evaluation function again to compute second-order expansion
	evaluations, _ = Secondorder_util(pi, mu, T, gamma, all_evals, V, trajs, rho, c, stop_grad=False)
	return evaluations


"""
High-level wrapper functions for computing estimates as differentiable parameters
used in the main loop
"""
def evaluations_vtrace(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for Vtrace evaluation

	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = jax.nn.softmax(params, -1)
	evals = Vtrace_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals
 

def evaluations_firstorder(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for first-order evaluation

	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = jax.nn.softmax(params, -1)
	evals = Firstorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals


def evaluations_secondorder(params, mu, T, gamma, V, trajs, rho, c):
	"""
	Compute evaluation output as a differentiable function of target policy parameter
	for second-order evaluation

	Args:
		params: target policy parameters
		mu: behavior policy
		T: length of partial trajectories
		gamma: discount factor
		V: bootstrapped value functions
		trajs: list of trajectories
		rho: truncation coefficient for IS ratio
		c: truncation coefficient for IS ratio
	Returns:
		estimates as a differentiable function of parameters
	"""
	pi = jax.nn.softmax(params, -1)
	evals = Secondorder_evaluation(pi, mu, T, gamma, V, trajs, rho, c)
	return evals


"""
Utility functions for computing exact values as oracle_values
"""
def policy_evaluation(P, R, discount, policy):
	"""
	Policy evaluation solver. Compute the exact values for a target policy.

	Args:
		P: transition matrix
		R: reward vector
		discount: discount factor
		policy: target policy
	Returns:
		Exact value function (vf) and Q-function (qf)
	"""
	nstates = P.shape[-1]
	ppi = jnp.einsum('ast,sa->st', P, policy)
	rpi = jnp.einsum('sa,sa->s', R, policy)
	vf = jnp.linalg.solve(np.eye(nstates) - discount*ppi, rpi)
	qf = R + discount*jnp.einsum('ast,t->sa', P, vf)
	return vf, qf


def get_PR(mdp):
	"""
	Extract transition matrix P and reward vector R from a mdp object

	Args:
		mdp: the MDP object
	Returns:
		The matrix P and vector R
	"""
	ns, na = mdp.ns, mdp.na
	P = np.zeros([na, ns, ns])
	R = np.zeros([ns, na])
	for i in range(na):
		for j in range(ns):
			P[i, j] = mdp.P[j * na + i]
	R = np.reshape(mdp.R_matrix, [ns, na])
	return P, R


def oracle_value(params, P, R, gamma):
	"""
	Compute the exact values for a target policy, at the initial state.
	The value is a differentiable function of the target policy parameters.

	Args:
		params: target policy parameters
		P: transition matrix
		R: reward vector
		gamma discount factor
	Returns:
		Exact value function at the initial state
	"""
	pi = jax.nn.softmax(params, -1)
	vf, _ = policy_evaluation(P, R, gamma, pi)
	return vf[0]