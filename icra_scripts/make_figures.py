# Created by William Edwards (wre2@illinois.edu)

# Standard project includes
import sys
from numbers import Number
from pdb import set_trace
import matplotlib

# External project includes
import numpy as np
import matplotlib.pyplot as plt

# Internal project includes
from utils import *

def make_figure_sysid1():
    models = [("ARX", "arx"), ("Koopman", "koop"),
            ("SINDy", "sindy"), ("MLP", "mlp")]
    tasks = [("Pendulum swing-up", "pendulum-swingup"),
            ( "Cartpole swing-up", "cartpole-swingup"),
            ("HalfCheetah", "halfcheetah")]
    settings = [
            ["cartpole-swingup", "arx", 10, 42],
            ["cartpole-swingup", "mlp", 100, 42],
            ["cartpole-swingup", "koop", 40, 42],
            ["cartpole-swingup", "sindy", 100, 42],
            ["cartpole-swingup", "approxgp", 100, 42],
            ["pendulum-swingup", "arx", 10, 42],
            ["pendulum-swingup", "mlp", 100, 42],
            ["pendulum-swingup", "koop", 40, 42],
            ["pendulum-swingup", "sindy", 100, 100],
            ["pendulum-swingup", "sindy", 100, 101],
            ["pendulum-swingup", "sindy", 100, 102],
            ["pendulum-swingup", "sindy", 100, 103],
            ["pendulum-swingup", "sindy", 100, 104],
            ["pendulum-swingup", "sindy", 100, 105],
            ["pendulum-swingup", "sindy", 100, 106],
            ["pendulum-swingup", "sindy", 100, 107],
            ["pendulum-swingup", "sindy", 100, 108],
            ["pendulum-swingup", "sindy", 100, 109],
            # ["cartpole-swingup", "sindy", 100, 100],
            # ["cartpole-swingup", "sindy", 100, 101],
            # ["cartpole-swingup", "sindy", 100, 102],
            # ["cartpole-swingup", "sindy", 100, 103],
            # ["cartpole-swingup", "sindy", 100, 104],
            # ["cartpole-swingup", "sindy", 100, 105],
            # ["cartpole-swingup", "sindy", 100, 106],
            # ["cartpole-swingup", "sindy", 100, 107],
            # ["cartpole-swingup", "sindy", 100, 108],
            # ["cartpole-swingup", "sindy", 100, 109],
            ["cartpole-swingup", "approxgp", 100, 42],
            ["halfcheetah", "arx", 9, 42],
            ["halfcheetah", "mlp", 100, 42],
            ["halfcheetah", "koop", 40, 42],
            ["halfcheetah", "sindy", 40, 42],
            ["halfcheetah", "approxgp", 100, 42],
            ]
    print("SysID Figure")
    print("============")
    print("SystemID ", end="")
    for task_label, _ in tasks:
        print(" & " + task_label, end="") 
    print(r"\\")
    for model_label, model_id in models:
        print(f"{model_label:8} ", end="")
        for task_label, task_id in tasks:
            scores = []
            for setting in settings:
                if setting[0] == task_id and setting[1] == model_id:
                    if result_exists("sysid1", *setting):
                        final_score, _ = load_result("sysid1", *setting)
                        scores.append(final_score)
                        #print(f"& {final_score:8.2f} ", end="")
            if not scores:
                print("& ", end="")
            else:
                median_score = np.median(scores)
                low_score = np.percentile(scores, 25)
                high_score = np.percentile(scores, 75)
                print(f"& {median_score:.2f} ({low_score:.2f}-{high_score:.2f})", 
                        end="")
        print("")

def make_figure_dalk1():
    result = load_result("sysid1", "dalk", "mlp", 100, 42)
    costs = result[1]["inc_costs"]
    baseline = 0.0334935

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(list(range(len(costs))), costs)
    ax.plot([0, len(costs)], [baseline, baseline], "r--") 
    ax.set_xlabel("Tuning Iterations")
    ax.set_ylabel("Model Accuracy")
    ax.set_title("Tuning MLP on DALK data")
    ax.set_ylim([0.0, 0.05])
    ax.legend(["MLP", "AR4"])
    plt.show()

def make_figure_cost_tuning():
    setting = ("cartpole-swingup", "mlp-ilqr", 100, 42)
    result, baseline_res = load_result("cost_tuning", *setting)

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.set_title(f"Cost Tuning Performance")
    ax.set_xlabel("Tuning iterations")
    ax.set_ylabel("True Perf.")
    perfs = [cost for cost in result["inc_truedyn_costs"]]
    print(f"{perfs=}")
    ax.plot(perfs)
    ax.plot([0.0, len(perfs)], [baseline_res[1], baseline_res[1]], "k--")
    ax.legend(["Tuned Quad. Cost", "Untuned Perf. Metric"])
    plt.tight_layout()
    plt.show()

def make_figure_tuning1(plot_option=1):
    experiments = [
            #(("MLP-iLQR-Quad", "Pendulum"),
            # [("pendulum-swingup", "mlp-ilqr", 100, 100),
            #  ("pendulum-swingup", "mlp-ilqr", 100, 101),
            #  ("pendulum-swingup", "mlp-ilqr", 100, 102)]
            #     ),
            (("MLP-iLQR-Custom", "Half-cheetah (OR)"),
             [#("halfcheetah", "halfcheetah", 3, 100, 100),
              ("halfcheetah", "halfcheetah", 3, 100, 101),
              ("halfcheetah", "halfcheetah", 3, 100, 102),
              ("halfcheetah", "halfcheetah", 3, 100, 103),
              ("halfcheetah", "halfcheetah", 3, 100, 104),
              ("halfcheetah", "halfcheetah", 3, 100, 105),
              ("halfcheetah", "halfcheetah", 3, 100, 106),
              ("halfcheetah", "halfcheetah", 3, 100, 107),
              ("halfcheetah", "halfcheetah", 3, 100, 108),
              ("halfcheetah", "halfcheetah", 3, 100, 109),
              ]),
            (("MLP-iLQR-Quad", "Swimmer"),
             [("swimmer", "swimmer", 1, 100, 100),
              ("swimmer", "swimmer", 1, 100, 101),
              ("swimmer", "swimmer", 1, 100, 102),
              ("swimmer", "swimmer", 1, 100, 103),
              #("swimmer", "swimmer", 1, 100, 104),
              ("swimmer", "swimmer", 1, 100, 105),
              ("swimmer", "swimmer", 1, 100, 106),
              ("swimmer", "swimmer", 1, 100, 107),
              ("swimmer", "swimmer", 1, 100, 108),
              ("swimmer", "swimmer", 1, 100, 109)
              ]),
            (("MLP-iLQR-Quad", "Pendulum"),
             [("pendulum-swingup", "mlp-ilqr", 100, 100),
              ("pendulum-swingup", "mlp-ilqr", 100, 101),
              ("pendulum-swingup", "mlp-ilqr", 100, 102),
              ("pendulum-swingup", "mlp-ilqr", 100, 103),
              ("pendulum-swingup", "mlp-ilqr", 100, 104),
              ("pendulum-swingup", "mlp-ilqr", 100, 105),
              #("pendulum-swingup", "mlp-ilqr", 100, 106),
              ("pendulum-swingup", "mlp-ilqr", 100, 107),
              ("pendulum-swingup", "mlp-ilqr", 100, 108),
              ("pendulum-swingup", "mlp-ilqr", 100, 109)
              ]),
            (("MLP-iLQR-Quad", "Cart-pole"),
             [("cartpole-swingup", "mlp-ilqr", 100, 100),
              ("cartpole-swingup", "mlp-ilqr", 100, 101),
              ("cartpole-swingup", "mlp-ilqr", 100, 102),
              ("cartpole-swingup", "mlp-ilqr", 100, 103),
              ("cartpole-swingup", "mlp-ilqr", 100, 104),
              ("cartpole-swingup", "mlp-ilqr", 100, 105),
              ("cartpole-swingup", "mlp-ilqr", 100, 106),
              ("cartpole-swingup", "mlp-ilqr", 100, 107),
              ("cartpole-swingup", "mlp-ilqr", 100, 108),
              ("cartpole-swingup", "mlp-ilqr", 100, 109),
              ]
                 ),

            (("MLP-iLQR-Custom", "Half-cheetah"),
             [("halfcheetah", "halfcheetah", 100, 100),
              ("halfcheetah", "halfcheetah", 100, 101),
              #("halfcheetah", "halfcheetah", 100, 102),
              ("halfcheetah", "halfcheetah", 100, 103),
              ("halfcheetah", "halfcheetah", 100, 104),
              ("halfcheetah", "halfcheetah", 100, 105),
              ("halfcheetah", "halfcheetah", 100, 106),
              ("halfcheetah", "halfcheetah", 100, 107),
              ("halfcheetah", "halfcheetah", 100, 108),
              ("halfcheetah", "halfcheetah", 100, 109),
              ]),
            (("MLP-iLQR-Custom", "Half-cheetah (Multisine)"),
             [#("halfcheetah", "halfcheetah", 100, 100),
              #("halfcheetah", "halfcheetah", 100, 101),
              ("halfcheetah-multisine", "halfcheetah", 100, 102),
              ("halfcheetah-multisine", "halfcheetah", 100, 103),
              ("halfcheetah-multisine", "halfcheetah", 100, 104),
              ("halfcheetah-multisine", "halfcheetah", 100, 105),
              ("halfcheetah-multisine", "halfcheetah", 100, 106),
              ("halfcheetah-multisine", "halfcheetah", 100, 107),
              ("halfcheetah-multisine", "halfcheetah", 100, 108),
              ("halfcheetah-multisine", "halfcheetah", 100, 109),
              ]),
            (("MLP-iLQR-Custom (More Surr Iters)", "Half-cheetah"),
             [#("halfcheetah", "halfcheetah", 100, 100),
              ("halfcheetah", "halfcheetah", 2, 100, 101),
              ("halfcheetah", "halfcheetah", 2, 100, 102),
              ("halfcheetah", "halfcheetah", 2, 100, 103),
              ("halfcheetah", "halfcheetah", 2, 100, 104),
              ("halfcheetah", "halfcheetah", 2, 100, 105),
              ("halfcheetah", "halfcheetah", 2, 100, 106),
              ("halfcheetah", "halfcheetah", 2, 100, 107),
              ("halfcheetah", "halfcheetah", 2, 100, 108),
              ("halfcheetah", "halfcheetah", 2, 100, 109),
              ]),
            #(("MLP-iLQR-Custom", "Half-cheetah"),
            # [("halfcheetah", "halfcheetah", 100, 100),
            #  ("halfcheetah", "halfcheetah", 100, 101)]
            #     )
            ]
            #(("MLP-iLQR", "Acrobot"),
            #    ("acrobot-swingup", "mlp-ilqr", 100, 42))
            #]
    #experiments = [
    #        (("MLP-iLQR-CustQuad", "Half-Cheetah"),
    #         ("halfcheetah", "mlp-ilqr", 100, 42)),
    #        ]
    #bcq_baselines = [73, 148, 200]
    bcq_baselines = [np.nan, 73, 148, 200, 200, 200]
    #bcq_baselines = [200]
    for i, ((pipeline_label, task_label), settings) in enumerate(experiments):
        #if not result_exists("tuning1", *setting):
        #    print(f"Skipping {pipeline_label}, {task_label}")
        #    continue
        results = [load_result("tuning1", *setting) for setting in settings]

        #matplotlib.rcParams.update({'font.size': 17})
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f"Tuning {task_label}")
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
        truedyn_perfss = [[cost for cost in result["inc_truedyn_costs"]] for result in results]
        set_trace()
        #perfs = [263.0] * 6 + [113.0]*4 + [535]*7 + [29]*68
        #print(f"{perfs=}")
        xs = list(range(len(truedyn_perfss[0])))
        if plot_option == 1:
            for truedyn_perfs in truedyn_perfss:
                ax.plot(truedyn_perfs)
        elif plot_option == 2:
            high = []
            middle = []
            low = []
            for j in range(len(truedyn_perfss[0])):
                perfs = [truedyn_perfs[j] for truedyn_perfs in truedyn_perfss]
                perfs.sort()
                low.append(min(perfs))
                middle.append(np.median(perfs))
                high.append(np.max(perfs))
            ax.plot(xs, middle, color="b")
            ax.fill_between(xs, low, high, color="b", alpha=0.2)
        elif plot_option == 3:
            high = []
            middle = []
            low = []
            for j in range(len(truedyn_perfss[0])):
                perfs = [truedyn_perfs[j] for truedyn_perfs in truedyn_perfss]
                perfs.sort()
                low.append(np.percentile(perfs, 25))
                middle.append(np.median(perfs))
                high.append(np.percentile(perfs, 75))
            ax.plot(xs, middle, color="b")
            ax.fill_between(xs, low, high, color="b", alpha=0.2)
        ax.plot([0, len(truedyn_perfss[0])], [bcq_baselines[i], bcq_baselines[i]], "r--") 
        ax.legend(["AutoMPC", "BCQ"])#, prop={"size":16})
        plt.tight_layout()
        plt.show()

def make_figure_cartpole_final():
    experiments = [
            (("MLP-iLQR", "Half-Cheetah"),
             ("halfcheetah", "mlp-ilqr", 100, 42)),
            ]
    #bcq_baselines = [24, 37, 1000]
    baselines = [83]
    experiments = [
            (("Tune SysID on Data", "Cartpole Swingup"),
                ("sysid2", "cartpole-swingup", 
                    "mlp-ilqr", 2, 100, 42)),
            (("Tune SysID on Perf.", "Cartpole Swingup"),
                ("sysid2", "cartpole-swingup", 
                    "mlp-ilqr", 3, 100, 42)),
            (("Tune Obj/Opt, Pre-tuned SysID", "Cartpole Swingup"),
                ("decoupled1", "cartpole-swingup", 
                    "mlp-ilqr", 100, 42)),
            (("Full Pipeline Tune", "Cartpole Swingup"),
                ("tuning1", "cartpole-swingup", 
                    "mlp-ilqr", 100, 42))
            ]

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(6,4))
    ax = fig.gca()
    ax.set_title("Tuning MLP-iLQR-Quad on Cartpole")
    ax.set_xlabel("Tuning iterations")
    ax.set_ylabel("True Dyn Perf.")
    labels = []
    for i, ((label1, label2), setting) in enumerate(experiments):
        #labels = []
        #for label, value in baselines:
        #    ax.plot([0.0, n_iters], [value, value], "--")
        #    labels.append(label)
        #for label, res in tuning_results:
        #    perfs = [-cost for cost in res["inc_truedyn_costs"]]
        #    ax.plot(perfs)
        #    labels.append(label)
        #ax.legend(labels)
        result = load_result(setting[0], *setting[1:])
        if isinstance(result, tuple):
            result = result[0]
        perfs = [cost for cost in result["inc_truedyn_costs"]]
        print(f"{perfs=}")
        ax.plot(perfs)
        labels.append(label1)
    ax.plot([0, len(perfs)], [baselines[0], baselines[0]], "r--") 
    ax.legend(labels + ["Hand-tuned Baseline"], prop={'size':11})
    plt.tight_layout()
    plt.show()


def make_figure_decoupled1():
    #result = load_result("decoupled1", "cartpole-swingup", "mlp-ilqr", 100,
    #        42)
    #result = load_result("decoupled1", "halfcheetah", "halfcheetah", 100, 43)
    #result = load_result("decoupled1", "halfcheetah", "halfcheetah", 100, 4, 44)
    #result = load_result("decoupled2_int", 80, 0, 1)
    result = load_result("ideal", "halfcheetah", "halfcheetah", 100, 8, 200)

    #matplotlib.rcParams.update({'font.size': 12})
    #fig = plt.figure(figsize=(4,4))
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(f"MLP-iLQR on Halfcheetah")
    ax.set_xlabel("Tuning iterations")
    ax.set_ylabel("True Perf.")
    ax.set_ylim([-275, 400])
    #bcq_baseline = 200
    #labels = []
    #for label, value in baselines:
    #    ax.plot([0.0, n_iters], [value, value], "--")
    #    labels.append(label)
    #for label, res in tuning_results:
    #    perfs = [-cost for cost in res["inc_truedyn_costs"]]
    #    ax.plot(perfs)
    #    labels.append(label)
    #ax.legend(labels)
    #perfs = [cost for cost in result["inc_costs"]]
    #perfs = [263.0] * 6 + [113.0]*4 + [535]*7 + [29]*25
    #print(f"{perfs=}")
    #ax.plot([c for c, _ in result[0]["inc_truedyn_costs"]])
    #ax.plot([s for _, (s,_,_,_) in result[0]["inc_truedyn_costs"]], "--")
    #ax.plot([s for _, (_,s,_,_) in result[0]["inc_truedyn_costs"]], "--")
    #ax.plot([s for _, (_,_,s,_) in result[0]["inc_truedyn_costs"]], "--")
    #ax.plot([s for _, (_,_,_,s) in result[0]["inc_truedyn_costs"]], "--")
    ax.plot([r[0] for r in result[0]["inc_truedyn_costs"]])
    ax.plot(result[0]["inc_costs"])
    #ax.plot([c for c, _ in result[0]["inc_costs"]])
    #ax.plot(result[0]["inc_truedyn_costs"])
    #ax.plot(result[0]["costs"])
    #ax.plot([0, len(perfs)], [37, 37], "r--") 
    #ax.plot([0, len(result[0]["costs"])], [bcq_baseline, bcq_baseline], "r--") 
    ax.legend(["True Dyn.", "Surrogate"])#, prop={"size":16})
    #ax.legend(["true cost", "surr. cost"])
    plt.tight_layout()
    plt.show()

def make_figure_decoupled2():
    root_seeds = [80, 81, 82]
    surr_idxs = [0, 1]
    smac_idxs = [0, 1]
    result = load_result("decoupled2_int", 80, 0, 1)

    fig, axs = plt.subplots(len(root_seeds), len(surr_idxs) * len(smac_idxs),
            figsize=(12,12))
    i = 0
    for seed in root_seeds:
        for surr_idx in surr_idxs:
            for smac_idx in smac_idxs:
                result = load_result("decoupled2_int", seed, surr_idx, smac_idx)
                ax = axs.flat[i]; i += 1
                #ax.set_aspect(0.1)
                ax.set_title("Seed {}, Surr {}, SMAC {}"
                        .format(seed, surr_idx, smac_idx))
                ax.set_xlabel("Tuning iterations")
                ax.set_ylabel("True Perf.")
                ax.set_ylim([-300, 400])
                ax.plot(result["inc_truedyn_costs"])
                ax.plot(result["inc_costs"])
                ax.legend(["true cost", "surr. cost"])
    plt.tight_layout()
    plt.show()

def pca_decoupled1():
    #result = load_result("decoupled1", "halfcheetah", "halfcheetah", 100, 44)
    result = load_result("decoupled2_int", 40, 0, 0)
    true_perfs = np.array(result["truedyn_costs"])
    cfgs = result["cfgs"]
    hyper_names = list(cfgs[0].get_dictionary().keys())
    hyper_vals = np.zeros((len(cfgs), len(hyper_names)))
    for i, cfg in enumerate(cfgs):
        for j, name in enumerate(hyper_names):
            hyper_vals[i, j] = cfg[name]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=5)
    pca.fit(hyper_vals, true_perfs)

    hyper_hr_names = ["horiz", "targvel", "u_bthigh_R", "u_bshin_R",
            "u_bfoot_R", "u_fthigh_R", "u_fhsin_R", "u_ffoot_R",
            "zpos_F", "zpos_Q", "fthigh_Q", "fshin_Q", "ffoot_Q",
            "xvel_F", "xvel_q"]
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(np.abs(pca.components_.T), cmap="Reds")
    comp_labels = ["{:.2f}%".format(100.0*ratio) 
            for i, ratio in enumerate(pca.explained_variance_ratio_)]
    plt.xticks(np.arange(0.0, len(comp_labels), 1), comp_labels)
    plt.yticks(np.arange(0.0, len(hyper_hr_names), 1), hyper_hr_names)
    plt.show()

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(true_perfs)
    ax.set_ylabel("Perf")
    ax2 = ax.twinx()
    ax2.set_ylabel("Horizon")
    ax2.plot(hyper_vals[:,0], color="r")
    #ax.legend(["Performance", "Horizon")
    plt.show()

    set_trace()


def make_figure_sysid2():
    setting1 = ("cartpole-swingup", "mlp-ilqr", 1, 100, 42)
    setting2 = ("cartpole-swingup", "mlp-ilqr", 2, 100, 42)
    setting3 = ("cartpole-swingup", "mlp-ilqr", 3, 100, 42)

    smac_res1, (rmses1, horizs1) = load_result("sysid2", *setting1)
    smac_res2, (rmses2, horizs2) = load_result("sysid2", *setting2)
    smac_res3, (rmses3, horizs3) = load_result("sysid2", *setting3)

    set_trace()

    matplotlib.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("RMSE")
    ax.set_title("Multi-Step Pred. Accuracy")
    ax.plot(horizs1, rmses1)
    ax.plot(horizs2, rmses2)
    ax.plot(horizs3, rmses3)
    ax.legend(["1-step train", "Multi-step train", "Pipeline train"])

    #fig = plt.figure(figsize=(4,4))
    #ax = fig.gca()
    #ax.set_xlabel("Tuning Iterations")
    #ax.set_ylabel("Performance")
    #ax.set_title("Pipeline Performance of Sys ID")
    #ax.plot(smac_res1["inc_truedyn_costs"])
    #ax.plot(smac_res2["inc_truedyn_costs"])
    #ax.plot(smac_res3["inc_truedyn_costs"])
    #ax.legend(["1-step train", "Multi-step train", "Pipeline train"])

    plt.tight_layout()
    plt.show()

def make_figure_surrtest():
    #setting = ("cartpole-swingup", "mlp-ilqr", 5, 42)
    true_scores =[201.0, 43.0, 49.0, 49.0, 125.0, 180.0, 39.0, 67.0] 
    surr_scoress = [
        [201.0, 201.0, 201.0, 201.0, 201.0, 201.0, 201.0, 201.0, 201.0, 201.0], 
        [39.0, 41.0, 43.0, 41.0, 43.0, 43.0, 44.0, 38.0, 43.0, 43.0], 
        [53.0, 51.0, 57.0, 54.0, 51.0, 49.0, 57.0, 50.0, 50.0, 56.0], 
        [53.0, 51.0, 57.0, 54.0, 51.0, 49.0, 57.0, 50.0, 50.0, 56.0], 
        [181.0, 155.0, 148.0, 161.0, 174.0, 150.0, 149.0, 133.0, 173.0, 141.0], 
        [201.0, 166.0, 201.0, 201.0, 201.0, 127.0, 201.0, 199.0, 201.0, 185.0], 
        [40.0, 39.0, 39.0, 38.0, 41.0, 39.0, 40.0, 41.0, 40.0, 40.0],
        [104.0, 54.0, 122.0, 56.0, 124.0, 120.0, 62.0, 118.0, 64.0, 130.0]
        ]
    medians = []
    errs = np.zeros((2, len(surr_scoress)))
    for i, surr_scores in enumerate(surr_scoress):
        surr_scores.sort()
        medians.append(surr_scores[5])
        errs[1, i] = surr_scores[7] - surr_scores[5]
        errs[0, i] = surr_scores[5] - surr_scores[2]


    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.set_title("Surr. vs True for MLP-iLQR-Quad on Cart-Pole")
    ax.set_xlabel("Surrogate Perf")
    ax.set_ylabel("True Perf")
    ax.errorbar(medians, true_scores, xerr=errs, fmt="yo", ecolor="k",
        capsize=2)

    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], "--", color="grey")

    plt.tight_layout()
    plt.show()

def make_figure_hyper_vals():
    seed = int(sys.argv[2])
    #result = load_result("decoupled1", "halfcheetah", "halfcheetah", 100, seed)
    result = load_result("decoupled1", "halfcheetah", "halfcheetah", 100, 4, seed)
    cfgs = result[0]["cfgs"]
    cs = cfgs[0].configuration_space
    for hp in cs.get_hyperparameters():
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(hp.name + " (" + str(seed) + ")")
        ax.set_xlabel("Tuning iteration")
        hp_vals = []
        for cfg in cfgs:
            if hp.name in cs.get_active_hyperparameters(cfg):
                val = cfg[hp.name]
                if isinstance(val, Number):
                    hp_vals.append(val)
                else:
                    hp_vals.append(hp.choices.index(val))
            else:
                hp_vals.append(np.nan)
        ax.plot(hp_vals)
        plt.show()

def main(command):
    if command == "sysid1":
        make_figure_sysid1()
    elif command == "dalk1":
        make_figure_dalk1()
    elif command == "tuning1":
        make_figure_tuning1()
    elif command == "sysid2":
        make_figure_sysid2()
    elif command == "cost_tuning":
        make_figure_cost_tuning()
    elif command == "decoupled":
        make_figure_decoupled1()
    elif command == "decoupled2":
        make_figure_decoupled2()
    elif command == "decoupled-pca":
        pca_decoupled1()
    elif command == "cartpole-final":
        make_figure_cartpole_final()
    elif command == "surrtest":
        make_figure_surrtest()
    elif command == "hypervals":
        make_figure_hyper_vals()
    else:
        raise Exception("Unrecognized command")

if __name__ == "__main__":
    main(sys.argv[1])
