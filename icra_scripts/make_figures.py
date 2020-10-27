# Created by William Edwards (wre2@illinois.edu)

# Standard project includes
import sys

# External project includes
import numpy as np
import matplotlib.pyplot as plt

# Internal project includes
from utils import *

def make_figure_sysid1():
    models = [("ARX", "arx"), ("Koopman", "koop"),
            ("SINDy", "sindy"), ("MLP", "mlp")]
    tasks = [("Pendulum swing-up", "pendulum-swingup"),
            ( "Cartpole swing-up", "cartpole-swingup")]
    settings = [
            ["cartpole-swingup", "arx", 10, 42],
            ["cartpole-swingup", "mlp", 100, 42],
            ["cartpole-swingup", "koop", 40, 42],
            ["cartpole-swingup", "sindy", 100, 42],
            ["cartpole-swingup", "approxgp", 100, 42],
            ["pendulum-swingup", "arx", 10, 42],
            ["pendulum-swingup", "mlp", 100, 42],
            ["pendulum-swingup", "koop", 40, 42],
            ["pendulum-swingup", "sindy", 100, 42],
            ["pendulum-swingup", "approxgp", 100, 42]]
    print("SysID Figure")
    print("============")
    print("SystemID ", end="")
    for task_label, _ in tasks:
        print(" & " + task_label, end="") 
    print(r"\\")
    for model_label, model_id in models:
        print(f"{model_label:8} ", end="")
        for task_label, task_id in tasks:
            for setting in settings:
                if setting[0] == task_id and setting[1] == model_id:
                    if result_exists("sysid1", *setting):
                        final_score, _ = load_result("sysid1", *setting)
                        print(f"& {final_score:8.2f} ", end="")
                        break
            else:
                print("&          ", end="")
        print(r" \\")

def make_figure_tuning1():
    experiments = [
            (("MLP-iLQR", "Pendulum Swing-up"),
             ("pendulum-swingup", "mlp-ilqr", 3, 42))
            ]
    for (pipeline_label, task_label), setting in experiments:
        if not result_exists("tuning1", *setting):
            print(f"Skipping {pipeline_label}, {task_label}")
            continue
        result = load_result("tuning1", *setting)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f"Tuning {pipeline_label} on {task_label}")
        ax.set_xlabel("Tuning iterations")
        ax.set_ylabel("True Perf.")
        #labels = []
        #for label, value in baselines:
        #    ax.plot([0.0, n_iters], [value, value], "--")
        #    labels.append(label)
        #for label, res in tuning_results:
        #    perfs = [-cost for cost in res["inc_truedyn_costs"]]
        #    ax.plot(perfs)
        #    labels.append(label)
        #ax.legend(labels)
        perfs = [cost for cost in result["inc_truedyn_costs"]]
        print(f"{perfs=}")
        ax.plot(perfs)
    plt.show()

def main(command):
    if command == "sysid1":
        make_figure_sysid1()
    elif command == "tuning1":
        make_figure_tuning1()
    else:
        raise Exception("Unrecognized command")

if __name__ == "__main__":
    main(sys.argv[1])
