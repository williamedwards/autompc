import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc

from scipy.integrate import solve_ivp

simplesys = ampc.System(["x1", "x2"], ["u"])

A = np.array([[1, -1], [0, 1]])
B = np.array([[0], [1]])

def sim_simplesys(x0, us):
    traj = ampc.zeros(simplesys, len(us) + 1)
    traj[0].obs[:] = x0

    for i, u in enumerate(us):
        traj[i, "u"] = u
        traj[i+1].obs[:] = A @ traj[i].obs + B @ [u]

    return traj

rng = np.random.default_rng(42)
samples = 100
length = 30

trajs = []
for _ in range(samples):
    x0 = rng.uniform(-10, 10, 2)
    us = rng.uniform(-1, 1, length)
    traj = sim_simplesys(x0, us)
    trajs.append(traj)

traj = trajs[-1]


print(traj.obs)
# Outputs:
## [[0.  0. ]
##  [0.  0. ]
##  [0.  0. ]
##  [0.  0. ]
##  [0.  0. ]
##  [0.5 0. ]
##  [3.  4. ]
##  [0.  0. ]
##  [0.  0. ]
##  [0.  0. ]]

print(traj.ctrls)
# Outputs:
## [[0.]
##  [0.]
##  [0.]
##  [1.]
##  [0.]
##  [0.]
##  [0.]
##  [0.]
##  [0.]
##  [0.]]


#from autompc.sysid import ARX, Koopman
#
#arx = ARX(pendulum)
#
#print(arx.is_linear)
## Ouputs:
### True
#
#print(arx.get_hyper_options())
## Ouputs:
### {'k': (<HyperType.int_range: 2>, (1, 10))}
#
#print(arx.get_hypers())
## Outputs:
### {'k': 1}
#
#arx.set_hypers(method="lasso")
#print(arx.get_hypers())
## Outputs:
### {'k': 5}
