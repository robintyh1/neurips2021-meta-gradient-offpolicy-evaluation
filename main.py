import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import os
import tabular_mdp
from jax import jacfwd, jacrev
from evaluation_utils import get_PR, oracle_value
from evaluation_utils import evaluations_vtrace, evaluations_firstorder, evaluations_secondorder


def generate_trajectories(mdp, mu, num_simulations, T):
	"""
	Generate trajectories from the MDP using policy mu

	Args:
		mdp: the mdp object
		mu: the policy to be executed
		num_simulations: num of trajectories
		T: truncated horizon of the trajectory
	Returns:
		A list of trajectories
	"""
	trajs = []
	na = mu.shape[1]
	for i in range(num_simulations):
		rsum = []
		states = []
		actions = []
		s = mdp.reset()
		for t in range(T):
			a = np.random.choice(np.arange(na), p=mu[s])
			s_next, r, _, _ = mdp.step(a)
			rsum.append(r)
			actions.append(a)
			states.append(s)
			s = s_next
		states.append(s_next)
		trajs.append({'states': states, 'actions': actions, 'rewards': rsum})
	return trajs


if __name__ == '__main__':
	# --------------------
	# Adjust hyper-parameters here
	# --------------------
	num_simulations = 1000  # num of trajectories used for averaging estimates
	epsilon = 0.8  # off-policyness
	independent_trials = 10  # num of independent trials
	ns, na = 20, 5  # state and action space dimension
	gamma = 0.8  # discount factor
	T = 20  # trajectory length
	density = 0.001  # parameter for generating MDP
	rho = np.inf  # truncation hyper-parameter -- default to no truncation
	c = np.inf  # truncation hyper-parameter -- default to no truncation

	# --------------------
	# Create directory to save data
	# --------------------
	directory = 'hessian_and_gradient_offpolicy_{}_eps{}'.format(num_simulations, epsilon)
	if not os.path.exists(directory):
		os.makedirs(directory)

	# --------------------
	# Create list to hold computed data
	# --------------------
	nocritic_grad = []
	nocritic_hessian = []
	truncated_grad = []
	truncated_hessian = []
	firstorder_grad = []
	firstorder_hessian = []
	loadeddice_grad = []
	loadeddice_hessian = []
	secondorder_grad = []
	secondorder_hessian = []

	# --------------------
	# Jit subroutines to speed up runtime computations
	# --------------------
	# jit Vtrace
	get_gradient_vtrace = jit(grad(evaluations_vtrace))
	get_hessian_vtrace = jit(jacfwd(jacrev(evaluations_vtrace)))

	# jit first-order
	get_gradient_firstorder = jit(grad(evaluations_firstorder))
	get_hessian_firstorder = jit(jacfwd(jacrev(evaluations_firstorder)))

	# jit second-order
	get_gradient_secondorder = jit(grad(evaluations_secondorder))
	get_hessian_secondorder = jit(jacfwd(jacrev(evaluations_secondorder)))

	# jit oracle evaluation
	oracle_gradient = jit(grad(oracle_value))
	oracle_hessian = jit(jacfwd(jacrev(oracle_value)))

	# --------------------
	# Main loop: loop through all evaluation methods
	# --------------------
	for trialite in range(independent_trials):

		# generate random MDP
		mdp = tabular_mdp.TabularMDP(ns, na, r_std=0.0, dirichlet_intensity=density)

		# construct target policy pi and behavior policy mu
		mu = np.ones([ns, na]) / float(na)
		pi = np.zeros([ns, na])
		pi[np.arange(ns), np.random.randint(0, na, ns)] = 1.
		pi = pi * epsilon + mu * (1-epsilon)

		# params for target policy
		params = np.log(pi)

		# Compute noise corrupted Vtrace values as bootstrapped values
		V_exact = mdp.evaluate(gamma, mu)['v']
		noise_level = 1.
		V_exact += np.random.randn(*np.shape(V_exact)) * noise_level * np.linalg.norm(V_exact) / V_exact.size
		V_bootstrapped = V_exact.copy()

		# get P, R from the MDP and compute oracle gradient and Hessian
		P, R = get_PR(mdp)
		oracle_gradient_value = oracle_gradient(params, P, R, gamma)
		oracle_hessian_value = oracle_hessian(params, P, R, gamma)
		
		# generate trajectories from MDP using behavior policy
		trajs_all = [generate_trajectories(mdp, mu, 1, T) for _ in range(num_simulations)]

		def corr(x, y):
			"""
			Angular accuracy measure between tensors
			between -1 and 1, the higher the better
			
			Args:
				two tensors x and y
			Returns:
				Angular accuracy measure
			"""
			x = x.flatten()
			y = y.flatten()
			x -= np.mean(x)
			y -= np.mean(y)
			return x.dot(y) / np.sqrt(x.dot(x) * y.dot(y))

		# --------------------
		# Evaluation based on importance sampling without bootstrapped value (no variance reduction)
		# --------------------
		print('step-wise IS (DiCE) with no critic')
		grads_all = np.zeros_like(params)
		hessians_all = None
		for i in range(num_simulations):
			trajs = trajs_all[i]
			grad_np = get_gradient_vtrace(params, mu, T, gamma, V_bootstrapped * 0.0, trajs, rho, c)
			hessian = get_hessian_vtrace(params, mu, T, gamma, V_bootstrapped * 0.0, trajs, rho, c)
			if hessians_all is None:
				hessians_all = np.zeros_like(hessian)
			grads_all += grad_np / num_simulations
			hessians_all += hessian / num_simulations
		grad_corr, hessian_corr = corr(grads_all, oracle_gradient_value), corr(hessians_all, oracle_hessian_value)
		print('gradient acc', grad_corr)
		print('hessian acc', hessian_corr)
		nocritic_grad.append(grad_corr)
		nocritic_hessian.append(hessian_corr)

		# --------------------
		# Evaluation based on doubly-robust method (importance sampling + bootstrapped value)
		# We also call it the "LoadedDice" method here
 		# --------------------
		print('loadeddice (doubly-robust)')
		grads_all = np.zeros_like(params)
		hessians_all = None
		for i in range(num_simulations):
			trajs = trajs_all[i]
			grad_np = get_gradient_vtrace(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			hessian = get_hessian_vtrace(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			if hessians_all is None:
				hessians_all = np.zeros_like(hessian)
			grads_all += grad_np / num_simulations
			hessians_all += hessian / num_simulations
		grad_corr, hessian_corr = corr(grads_all, oracle_gradient_value), corr(hessians_all, oracle_hessian_value)
		print('gradient acc', grad_corr)
		print('hessian acc', hessian_corr)
		loadeddice_grad.append(grad_corr)
		loadeddice_hessian.append(hessian_corr)

		# --------------------
		# Evaluation based on Vtrace, also named "Truncated importance sampling"
		# truncate the IS ratio at value of 2.0
		# --------------------
		print('vtrace')
		grads_all = np.zeros_like(params)
		hessians_all = None
		ctruncated = 1.0
		rhotruncated = 1.0
		for i in range(num_simulations):
			trajs = trajs_all[i]
			grad_np = get_gradient_vtrace(params, mu, T, gamma, V_bootstrapped, trajs, rho=rhotruncated, c=ctruncated)
			hessian = get_hessian_vtrace(params, mu, T, gamma, V_bootstrapped, trajs, rho=rhotruncated, c=ctruncated)
			if hessians_all is None:
				hessians_all = np.zeros_like(hessian)
			grads_all += grad_np / num_simulations
			hessians_all += hessian / num_simulations
		grad_corr, hessian_corr = corr(grads_all, oracle_gradient_value), corr(hessians_all, oracle_hessian_value)
		print('gradient acc', grad_corr)
		print('hessian acc', hessian_corr)
		truncated_grad.append(grad_corr)
		truncated_hessian.append(hessian_corr)

		# --------------------
		# Evaluation based on first-order expansion of value function
		# --------------------
		print('first-order')
		grads_all = np.zeros_like(params)
		hessians_all = None
		for i in range(num_simulations):
			trajs = trajs_all[i]
			grad_np = get_gradient_firstorder(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			hessian = get_hessian_firstorder(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			if hessians_all is None:
				hessians_all = np.zeros_like(hessian)
			grads_all += grad_np / num_simulations
			hessians_all += hessian / num_simulations
		grad_corr, hessian_corr = corr(grads_all, oracle_gradient_value), corr(hessians_all, oracle_hessian_value)
		print('gradient acc', grad_corr)
		print('hessian acc', hessian_corr)
		firstorder_grad.append(grad_corr)
		firstorder_hessian.append(hessian_corr)

		# --------------------
		# Evaluation based on second-order expansion of value function
 		# --------------------
		print('second-order')
		grads_all = np.zeros_like(params)
		hessians_all = None
		for i in range(num_simulations):
			trajs = trajs_all[i]
			grad_np = get_gradient_secondorder(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			hessian = get_hessian_secondorder(params, mu, T, gamma, V_bootstrapped, trajs, rho, c)
			if hessians_all is None:
				hessians_all = np.zeros_like(hessian)
			grads_all += grad_np / num_simulations
			hessians_all += hessian / num_simulations
		grad_corr, hessian_corr = corr(grads_all, oracle_gradient_value), corr(hessians_all, oracle_hessian_value)
		print('1st', grad_corr)
		print('2nd', hessian_corr)
		secondorder_grad.append(grad_corr)
		secondorder_hessian.append(hessian_corr)		

		# --------------------
		# Save all data to the directory
 		# --------------------
		np.save(directory + '/nocritic_grad', nocritic_grad)
		np.save(directory + '/nocritic_hessian', nocritic_hessian)
		np.save(directory + '/truncated_grad', truncated_grad)
		np.save(directory + '/truncated_hessian', truncated_hessian)
		np.save(directory + '/firstorder_grad', firstorder_grad)
		np.save(directory + '/firstorder_hessian', firstorder_hessian)
		np.save(directory + '/loadeddice_grad', loadeddice_grad)
		np.save(directory + '/loaddedice_hessian', loadeddice_hessian)
		np.save(directory + '/secondorder_grad', secondorder_grad)
		np.save(directory + '/secondorder_hessian', secondorder_hessian)