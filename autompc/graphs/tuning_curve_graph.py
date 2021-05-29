from ..tuning.pipeline_tuner import PipelineTuneResult
from ..tuning.model_tuner import ModelTuneResult

class TuningCurveGraph:
    def __call__(self, ax, tune_result):
        if isinstance(tune_result, PipelineTuneResult):
            if tune_result.inc_truedyn_costs is not None:
                ax.plot(tune_result.inc_truedyn_costs, label="True Dyn. Cost") 
            ax.plot(tune_result.inc_costs, label="Surr. Cost")
            ax.set_xlabel("Tuning Iteration")
            ax.set_ylabel("Cost")
            ax.legend()
        elif isinstance(tune_result, ModelTuneResult):
            ax.plot(tune_result.inc_costs, label="Surr. Cost")
            ax.set_xlabel("Tuning Iteration")
            ax.set_ylabel("Model Error")
