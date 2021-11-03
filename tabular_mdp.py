import numpy as np


def density(density_intensity, ns):
	a = np.random.dirichlet(np.ones(ns) * density_intensity)
	return a / np.sum(a)


class TabularMDP(object):
	"""
	A tabular MDP with gym-like interface
	"""

	def __init__(self, ns, na, r_std=0.1, r_mean_std=1.0, dirichlet_intensity=1.0):
		self.ns = ns
		self.na = na

		# transition matrix
		self.P = np.array([density(dirichlet_intensity, ns) for _ in range(ns*na)])

		# mean reward
		self.r_mean = {i: np.random.rand(na) * r_mean_std for i in range(ns)}

		# fixed initial state
		self.s = 0

		# convert to matrix quantity
		self.R_matrix = np.zeros([ns * na])
		for i in range(ns):
			for j in range(na):
				self.R_matrix[i * na + j] = self.r_mean[i][j]

	def reset(self):
		self.s = 0
		return self.s

	def step(self, action):
		# next state
		next_s_prob = self.P[self.s * self.na + action]
		next_s = np.random.choice(np.arange(self.ns), p=next_s_prob)
		
		# reward
		r = self.r_mean[self.s][action]

		# update
		self.s = next_s

		return self.s, r, False, {}

	def evaluate(self, gamma, pi):
		"""
		Construct exact evaluation quantities
		"""

		# assume pi to be a policy np,array
		assert pi.shape == (self.ns, self.na)
		ns, na = self.ns, self.na

		# transition matrix and reward matrix
		P_pi_matrix = np.zeros([ns * na, ns * na])
		for i in range(ns):
			for j in range(na):
				for i_prime in range(ns):
					for j_prime in range(na):
						P_pi_matrix[i * na + j, i_prime * na + j_prime] = self.P[i * self.na + j][i_prime] * pi[i_prime, j_prime]
		R = self.R_matrix

		# compute Q-function
		Q_matrix = np.linalg.inv(np.eye(ns * na) - gamma * P_pi_matrix).dot(R)

		# value function
		Q = Q_matrix.reshape([ns, na])
		V = np.sum(Q * pi, axis=-1)

		return {'v': V, 'q': Q}