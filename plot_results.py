import matplotlib.pyplot as plt
import numpy as np


eps_list = [0.0, 0.2, 0.5, 0.8]
num_simulations = 1000


data_mean = []
data_std = []
for eps in eps_list:
	d = np.load('hessian_and_gradient_offpolicy_{}_eps{}/loaddedice_hessian.npy'.format(num_simulations, eps))
	data_mean.append(np.mean(d))
	data_std.append(np.std(d))
h1 = plt.errorbar(eps_list, data_mean, data_std, color='m')


data_mean = []
data_std = []
for eps in eps_list:
	d = np.load('hessian_and_gradient_offpolicy_{}_eps{}/nocritic_hessian.npy'.format(num_simulations, eps))
	data_mean.append(np.mean(d))
	data_std.append(np.std(d))
h2 = plt.errorbar(eps_list, data_mean, data_std, color='g')


data_mean = []
data_std = []
for eps in eps_list:
	d = np.load('hessian_and_gradient_offpolicy_{}_eps{}/firstorder_hessian.npy'.format(num_simulations, eps))
	data_mean.append(np.mean(d))
	data_std.append(np.std(d))
h3 = plt.errorbar(eps_list, data_mean, data_std, color='r')


data_mean = []
data_std = []
for eps in eps_list:
	d = np.load('hessian_and_gradient_offpolicy_{}_eps{}/secondorder_hessian.npy'.format(num_simulations, eps))
	data_mean.append(np.mean(d))
	data_std.append(np.std(d))
h4 = plt.errorbar(eps_list, data_mean, data_std, color='b')


plt.xlim((-.05, 1.05))
plt.xlabel('Off-policyness')
plt.ylabel('Correlations')
plt.legend([h1, h2, h3, h4], ['doubly-robust (loaded DiCE)', 'stepwise IS (DiCE)', 'first-order', 'second-order'])
plt.show()