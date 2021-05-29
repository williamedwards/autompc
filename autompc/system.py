# Created by William Edwards (wre2@illinois.edu)

class System:
    def __init__(self, observations, controls):
        # Check inputs
        obs_set = set(observations)
        ctrl_set = set(controls)
        e = ValueError("Observation and control labels must be unique")
        if len(obs_set) != len(observations):
            raise e
        if len(ctrl_set) != len(controls):
            raise e
        if ctrl_set.intersection(obs_set):
            raise e

        self._controls = controls[:]
        self._observations = observations[:]

    def __eq__(self, other):
        return ((self.controls == other.controls) 
                and (self.observations == other.observations))

    @property
    def controls(self):
        return self._controls[:]

    @property
    def observations(self):
        return self._observations[:]

    @property
    def ctrl_dim(self):
        return len(self._controls)

    @property
    def obs_dim(self):
        return len(self._observations)
