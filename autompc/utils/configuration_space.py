from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError

class PipelineConfigurationSpace(ConfigurationSpace):
    def __init__(self, pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = pipeline

    def _check_forbidden(self, vector):
        super()._check_forbidden(vector)
        cfg = Configuration(self, vector=vector)
        if not self.pipeline.is_cfg_compatible(cfg):
            raise ForbiddenValueError("Pipeline incompatible")