# Created by William Edwards (wre2@illinois.edu)

# Standard project includes
import sys

# External project includes
import numpy as np
import matplotlib.pyplot as plt

# Internal project includes
from utils import *

def make_figure_sysid1():
    models = [("ARX", "arx"), ("Koopman", "koop")]
    tasks = [("Cartpole swing-up", "cartpole-swingup"),
            ("Pendulum swing-up", "pendulum-swingup")]
    settings = [["cartpole-swingup", "arx", 1, 42],
            ["cartpole-swingup", "koop", 1, 42]]
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


def main(command):
    if command == "sysid1":
        make_figure_sysid1()
    else:
        raise Exception("Unrecognized command")

if __name__ == "__main__":
    main(sys.argv[1])
