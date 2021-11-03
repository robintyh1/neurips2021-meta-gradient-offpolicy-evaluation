# Unifying Gradient Estimators for Meta-Reinforcement Learning via Off-Policy Evaluation @ NeurIPS 2021

This is the open source implementation of the toy example in the [NeurIPS 2021 paper](https://arxiv.org/abs/2106.13125). 

In the toy example, we examine the property of a few gradient and Hessian estimates of value functions in the tabular MDP. These estimates are used as subroutines for meta RL gradient estimates.

Installation
------------------
You need to install [JAX](https://github.com/google/jax). Our code works under Python 3.8 and you can install JAX by running the following
```bash
pip install jax
pip install jaxlib
```

Introduction to the code structure
------------------
The code contains a few components.

- `main.py` implements the main loop for the experiments. It creates MDP instances, generates trajectories and computes estimates and their accuracy measures.
- `evaluation_utils.py` implements different estimates through off-policy evaluation subroutines.
- `tabular_mdp.py` implements the tabular MDP object.
- `plot_results.py` plots the results similar to Fig 1 in the paper.

A few important aspects of the implementation:

- `evaluation_utils.py` implements a number of off-policy evaluation subroutines, such as [V-trace](https://arxiv.org/abs/1802.01561), [first-order](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) and [second-order value function expansions](https://arxiv.org/pdf/2003.06259.pdf).
- [Doubly-robust](https://arxiv.org/abs/1511.03722) and [step-wise importance sampling](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs) can be implemented as special cases of aforementioned methods.
- Prior methods such as [DiCE](https://arxiv.org/abs/1802.05098), its follow-up [variants](https://arxiv.org/abs/1909.10549), and [LVC](https://openreview.net/pdf?id=SkxXCi0qFX) are instantiated by the above methods. See Table 1 in the paper for details.

Running the code
------------------
To run all experiments, run the following. Note that you can directly adjust hyper-parameters in `main.py`
```bash
python main.py
```

After the experiments terminate, run the following to plot results.
```bash
python plot_results.py
```

Citation
------------------
If you find this code base useful, you are encouraged to cite the following paper

```
@article{tang2021unifying,
  title={Unifying Gradient Estimators for Meta-Reinforcement Learning via Off-Policy Evaluation},
  author={Tang, Yunhao and Kozuno, Tadashi and Rowland, Mark and Munos, R{\'e}mi and Valko, Michal},
  journal={arXiv preprint arXiv:2106.13125},
  year={2021}
}
```
