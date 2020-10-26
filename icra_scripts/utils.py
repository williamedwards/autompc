# Created by Willliam Edwards (wre2@illinois.edu), 2020-10-26

# Standard library includes
import os, sys
import pickle

# Constants
save_path = "results"


def save_result(result, exp_name, *attrs):
    file_name = exp_name + "".join(["_" + str(attr) for attr in attrs]) + ".pkl"
    file_path = os.path.join(save_path, file_name)
    print(f"{file_path=}")
    with open(file_path, "wb") as f:
        pickle.dump(result, f)

def load_result(exp_name, *attrs):
    file_name = exp_name + "".join(["_" + str(attr) for attr in attrs]) + ".pkl"
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "rb") as f:
        result = pickle.load(f)
    return result

def result_exists(exp_name, *attrs):
    file_name = exp_name + "".join(["_" + str(attr) for attr in attrs]) + ".pkl"
    file_path = os.path.join(save_path, file_name)
    return os.path.exists(file_path)
