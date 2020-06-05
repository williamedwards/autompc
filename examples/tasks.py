import numpy as np
import autompc as ampc

pendulum = ampc.System(["ang", "angvel"], ["torque"])

task = ampc.Task(pendulum)

Q = np.eye(2)
R = np.eye(1)

# Set costs
task.set_quad_cost(Q, R)

# Add constraints
task.set_state_bound("angvel", -1.0, 1.0)
task.set_ctrl_bound("torque", -0.5, 0.5)

print(task.is_qp)
# Desire output: True

# More complex options
def numerical_cost(traj):
    # Compute cost and grad
    return cost, grad
task.set_diff_cost(cost)

def free_space_constraint(traj):
    for i in range(len(traj)):
        if not traj[i].obs in free_space:
            return False
    return True

task.add_bool_cons(traj)

