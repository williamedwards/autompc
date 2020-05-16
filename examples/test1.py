import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc

pendulum = ampc.System(["ang", "angvel"], ["torque"])

traj = ampc.zeros(pendulum, 10)

traj[3, "torque"] = 1.0
traj[5, "ang"] = 0.5
traj[6].obs[:] = np.array([3.0, 4.0])

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


from autompc.sysid import ARX, Koopman

arx = ARX(pendulum)

print(arx.is_linear)
# Ouputs:
## True

print(arx.get_hyper_options())
# Ouputs:
## {'k': (<HyperType.int_range: 2>, (1, 10))}

print(arx.get_hypers())
# Outputs:
## {'k': 1}

arx.set_hypers(method="lasso")
print(arx.get_hypers())
# Outputs:
## {'k': 5}
