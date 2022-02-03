
import sys
import numpy as np
import autompc.sysid as sysid
import autompc.benchmarks as benchmarks
from autompc.evaluation import HoldoutModelEvaluator
from autompc.evaluation.model_metrics import get_model_rmse
from autompc.tuning import ModelTuner

def run_benchmark_test(model_class, benchmark_class):
    benchmark = benchmark_class()
    system = benchmark.system
    trajs = benchmark.gen_trajs(n_trajs=500, traj_len=200, seed=100)
    model = model_class(system)

    evaluator = HoldoutModelEvaluator(system, metric="rmse", holdout_prop=0.1, trajs=trajs,
                                    rng=np.random.default_rng(100))

    tuner = ModelTuner(system, evaluator)
    tuner.add_model(model)

    tuned_model, tune_result = tuner.run(n_iters=200, rng=np.random.default_rng(100))

    final_evaluation_trajs = benchmark.gen_trajs(n_trajs=2, traj_len=200, seed=100)
    model_score = get_modle_rmse(tuned_model, final_evaluation_trajs, horizon=20)

    print(f"Model Score = {model_score}")

def main(model_name, benchmark_name):
    model_class = getattr(sysid, model_name)
    benchmark_class = getattr(benchmarks, benchmark_name)
    run_benchmark_test(model_class, benchmark_class)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])