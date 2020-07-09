import sys, os

import numpy as np
import autompc as ampc
from pdb import set_trace

dummy = ampc.System(["x1", "x2"], ["u"])


from autompc.sysid.dummy_nonlinear import DummyNonlinear
x0 = [2.0, 3.0]
u0 = [1.0]
traj = ampc.zeros(dummy, 1)
traj[0].obs[:] = x0
traj[0].ctrl[:] = u0

model = DummyNonlinear(dummy)
xnew, latent, grad = model.pred_diff(traj)

print("xnew={}".format(xnew))

print("d x1[k+1] / d x2[k] = {}".format(grad[-1, "x2", 0]))
