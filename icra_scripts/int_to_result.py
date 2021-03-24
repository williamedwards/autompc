import sys, pickle
from pdb import set_trace

def main():
    in_fn = sys.argv[1]
    out_fn = sys.argv[2]
    
    costs = []
    truedyn_costs = []
    inc_costs = []
    inc_truedyn_costs = []
    surr_cost = None
    inc_cost = float("inf")
    inc_truedyn_cost = float("inf")
    with open(in_fn, "r") as f:
        for line in f:
            if "Surrogate score is" in line:
                surr_cost = float(line.split(" ")[-1])
            elif "True dynamics score is" in line:
                truedyn_cost = float(line.split(" ")[-1])
                costs.append(surr_cost)
                truedyn_costs.append(truedyn_cost)
                if surr_cost < inc_cost:
                    inc_cost = surr_cost
                    inc_truedyn_cost = truedyn_cost
                inc_costs.append(inc_cost)
                inc_truedyn_costs.append(inc_truedyn_cost)

    ret_value = dict()
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_truedyn_costs"] = inc_truedyn_costs
    ret_value["costs"] = costs
    ret_value["truedyn_costs"] = truedyn_costs

    with open(out_fn, "wb") as f:
        pickle.dump(ret_value, f)

if __name__ == "__main__":
    main()
