import sys, os
sys.path.append(os.getcwd() + "/..")

from pdb import set_trace

import autompc as ampc
from autompc.sysid.arx import ARX

pendulum = ampc.System(["ang", "angvel"], ["torque"])

print(pendulum.controls)
print(pendulum.observations)
print(pendulum.ctrl_dim)
print(pendulum.obs_dim)

#model = ampc.Model()
#arx = ARX()
#
#print(isinstance(arx, ampc.Model))
#print(arx.is_trainable)
