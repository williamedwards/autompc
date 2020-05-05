import sys, os
sys.path.append(os.getcwd() + "/..")

from pdb import set_trace

import numpy as np
import autompc as ampc
from autompc.sysid.arx import ARX

pendulum = ampc.System(["ang", "angvel"], ["torque"])

traj = ampc.zeros(pendulum, 10)

traj[3, "torque"] = 1.0
traj[5, "ang"] = 0.5
traj[6].obs[:] = np.array([3.0, 4.0])
print(traj.obs)
print(traj.ctrls)

print(traj[3, "torque"])
print(traj[5].obs)

arx = ARX()

print(arx.get_hyper_options())
print(arx.get_hypers())
arx.set_hypers(k=5)
print(arx.get_hypers())

