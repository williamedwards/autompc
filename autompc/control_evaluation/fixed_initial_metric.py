# Created by William Edwards (wre2@illinois.edu)
from pdb import set_trace

import numpy as np

from .control_metric import ControlMetric
from ..trajectory import zeros, extend
from func_timeout import func_timeout, FunctionTimedOut

class FixedInitialMetric(ControlMetric):
    def __init__(self, system, task, init_states, sim_iters,
            sim_time_limit=1.0, cost_limit=1e5):
        super().__init__(system, task)
        self.init_states = init_states
        self.sim_iters = sim_iters
        self.sim_time_limit = sim_time_limit
        self.cost_limit = cost_limit

    def __call__(self, controller, sim_model, ret_detailed=False):
        costs = []
        for init_state in self.init_states:
            try:
                sim_traj = func_timeout(self.sim_time_limit, self._run_sim, 
                        args=(controller, sim_model, init_state))
                cost = self.task.get_cost()(sim_traj)
                print(f"Last obs={sim_traj[-1].obs}\nCost={cost}\n***")
                if cost > self.cost_limit or np.isnan(cost):
                    cost = self.cost_limit
                costs.append(cost)
            except FunctionTimedOut:
                print("Simulation time out")
                costs.append(self.cost_limit)

        if ret_detailed:
            return np.mean(costs), costs
        else:
            return np.mean(costs)
        
    def _run_sim(self, controller, sim_model, init_state):
        sim_traj = zeros(self.system, 1)
        x = np.copy(init_state)
        sim_traj[0].obs[:] = x
        
        constate = controller.traj_to_state(sim_traj)
        simstate = sim_model.traj_to_state(sim_traj)
        for _  in range(self.sim_iters):
            u, constate = controller.run(constate, sim_traj[-1].obs)
            simstate = sim_model.pred(simstate, u)
            x = simstate[:self.system.obs_dim]
            sim_traj[-1, "u"] = u
            sim_traj = extend(sim_traj, [x], [[0.0]])
        return sim_traj

