import sys
import numpy as np
import torch
import joblib
from joblib import Memory
memory = Memory("cache")
import autompc as ampc
from ConfigSpace import Configuration

sys.path.append("../icra_scripts")

from pipelines import init_halfcheetah
from halfcheetah_task import halfcheetah_task
from decoupled1 import train_mlp, runsim
from pdb import set_trace

def get_cfgs(cs):
    pass
    cfg1 = cs.get_default_configuration()
    cfg1["_controller:horizon"] = 15
    cfg1["_task_transformer_0:target_velocity"] = 2.5
    cfg1["_task_transformer_0:u0_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:u1_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:u2_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:u3_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:u4_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:u5_log10Rgain"] = 0.5
    cfg1["_task_transformer_0:x1_log10Fgain"] = 0.5
    cfg1["_task_transformer_0:x1_log10Qgain"] = 0.5
    cfg1["_task_transformer_0:x6_log10Qgain"] = 0.5
    cfg1["_task_transformer_0:x7_log10Qgain"] = 0.5
    cfg1["_task_transformer_0:x8_log10Qgain"] = 0.5
    cfg1["_task_transformer_0:x9_log10Fgain"] = 0.5
    cfg1["_task_transformer_0:x9_log10Qgain"] = 0.5

    cfg2 = cs.get_default_configuration() # Line 639
    cfg2["_controller:horizon"] = 5
    cfg2["_task_transformer_0:target_velocity"] = 3.325140002322952
    cfg2["_task_transformer_0:u0_log10Rgain"] = 1.8225839635513923
    cfg2["_task_transformer_0:u1_log10Rgain"] = 0.7042124193310815
    cfg2["_task_transformer_0:u2_log10Rgain"] = -2.972987304908419
    cfg2["_task_transformer_0:u3_log10Rgain"] = -2.057988383065071
    cfg2["_task_transformer_0:u4_log10Rgain"] = -0.5192104514945992
    cfg2["_task_transformer_0:u5_log10Rgain"] = -0.3298195726072204
    cfg2["_task_transformer_0:x1_log10Fgain"] = -0.19684817855990389
    cfg2["_task_transformer_0:x1_log10Qgain"] = 2.915457035228112
    cfg2["_task_transformer_0:x6_log10Qgain"] = 0.305534713996384
    cfg2["_task_transformer_0:x7_log10Qgain"] = 0.9687839065627548
    cfg2["_task_transformer_0:x8_log10Qgain"] = 3.9353742693844893
    cfg2["_task_transformer_0:x9_log10Fgain"] = 0.6494969830078636
    cfg2["_task_transformer_0:x9_log10Qgain"] = 3.6998831191387502

    cfg3 = cs.get_default_configuration() # Line 617
    cfg3["_controller:horizon"] = 14
    cfg3["_task_transformer_0:target_velocity"] = 3.325140002322952
    cfg3["_task_transformer_0:u0_log10Rgain"] = 1.8225839635513923
    cfg3["_task_transformer_0:u1_log10Rgain"] = 0.9998168851884417
    cfg3["_task_transformer_0:u2_log10Rgain"] = -2.3628048944843876
    cfg3["_task_transformer_0:u3_log10Rgain"] = -1.726560617346112
    cfg3["_task_transformer_0:u4_log10Rgain"] = -0.25134435175288683
    cfg3["_task_transformer_0:u5_log10Rgain"] = -0.2814924123990199
    cfg3["_task_transformer_0:x1_log10Fgain"] = -0.19684817855990389
    cfg3["_task_transformer_0:x1_log10Qgain"] = 2.915457035228112
    cfg3["_task_transformer_0:x6_log10Qgain"] = 0.730786515018008
    cfg3["_task_transformer_0:x7_log10Qgain"] = 0.9687839065627548
    cfg3["_task_transformer_0:x8_log10Qgain"] = 3.9114350837257694
    cfg3["_task_transformer_0:x9_log10Fgain"] = 0.9606083792493711
    cfg3["_task_transformer_0:x9_log10Qgain"] = 3.727664867971069

    cfg4 = cs.get_default_configuration() # Line 683
    cfg4["_controller:horizon"] = 20
    cfg4["_task_transformer_0:target_velocity"] = 3.0809232468760555
    cfg4["_task_transformer_0:u0_log10Rgain"] = 1.6784007197171764
    cfg4["_task_transformer_0:u1_log10Rgain"] = 0.5623771457196196
    cfg4["_task_transformer_0:u2_log10Rgain"] = -2.972987304908419
    cfg4["_task_transformer_0:u3_log10Rgain"] = -2.057988383065071
    cfg4["_task_transformer_0:u4_log10Rgain"] = -0.5192104514945992
    cfg4["_task_transformer_0:u5_log10Rgain"] = -0.6768369000635972
    cfg4["_task_transformer_0:x1_log10Fgain"] = -0.19684817855990389
    cfg4["_task_transformer_0:x1_log10Qgain"] = 3.3890798708510976
    cfg4["_task_transformer_0:x6_log10Qgain"] = 1.7568456205117826
    cfg4["_task_transformer_0:x7_log10Qgain"] = 1.2478590880978997
    cfg4["_task_transformer_0:x8_log10Qgain"] = 3.9353742693844893
    cfg4["_task_transformer_0:x9_log10Fgain"] = -1.0563776161238354
    cfg4["_task_transformer_0:x9_log10Qgain"] = 3.571336964407556

    cfg5 = cs.get_default_configuration() # Line 1960
    cfg5["_controller:horizon"] = 18
    cfg5["_task_transformer_0:target_velocity"] = 2.525163027372873
    cfg5["_task_transformer_0:u0_log10Rgain"] = -0.04624389620409941
    cfg5["_task_transformer_0:u1_log10Rgain"] = 0.22493489841509273
    cfg5["_task_transformer_0:u2_log10Rgain"] = -2.9615128692809933
    cfg5["_task_transformer_0:u3_log10Rgain"] = -1.2035422092302555
    cfg5["_task_transformer_0:u4_log10Rgain"] = -1.9583279350235452
    cfg5["_task_transformer_0:u5_log10Rgain"] = -2.750222446265549
    cfg5["_task_transformer_0:x1_log10Fgain"] = 0.7271424840906588
    cfg5["_task_transformer_0:x1_log10Qgain"] = 3.1964066198574406
    cfg5["_task_transformer_0:x6_log10Qgain"] = 2.04610428622707
    cfg5["_task_transformer_0:x7_log10Qgain"] = 0.7143571876517938
    cfg5["_task_transformer_0:x8_log10Qgain"] = 3.378607443837293
    cfg5["_task_transformer_0:x9_log10Fgain"] = -0.9205515632546732
    cfg5["_task_transformer_0:x9_log10Qgain"] = 3.3964123503604933

    cfg6 = cs.get_default_configuration() # Line 507
    cfg6["_controller:horizon"] = 9
    cfg6["_task_transformer_0:target_velocity"] = 4.21875
    cfg6["_task_transformer_0:u0_log10Rgain"] = -1.46875
    cfg6["_task_transformer_0:u1_log10Rgain"] = 0.71875
    cfg6["_task_transformer_0:u2_log10Rgain"] = 3.34375
    cfg6["_task_transformer_0:u3_log10Rgain"] = 0.28125
    cfg6["_task_transformer_0:u4_log10Rgain"] = 3.34375
    cfg6["_task_transformer_0:u5_log10Rgain"] = 2.03125
    cfg6["_task_transformer_0:x1_log10Fgain"] = -1.90625
    cfg6["_task_transformer_0:x1_log10Qgain"] = 2.46875
    cfg6["_task_transformer_0:x6_log10Qgain"] = 2.46875
    cfg6["_task_transformer_0:x7_log10Qgain"] = -1.46875
    cfg6["_task_transformer_0:x8_log10Qgain"] = -1.46875
    cfg6["_task_transformer_0:x9_log10Fgain"] = -2.78125
    cfg6["_task_transformer_0:x9_log10Qgain"] = 1.59375

    return [cfg1, cfg2, cfg3, cfg4, cfg5, cfg6]

def main():
    tinf = halfcheetah_task()
    pipeline = init_halfcheetah(tinf)
    seed = 42
    subexp = 2
    new_surr_count = 10

    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))

    torch.manual_seed(rng.integers(1 << 30))
    surrogate = train_mlp(tinf.system, surr_trajs)
    eval_seed = rng.integers(1 << 30)
    print(f"{eval_seed=}")

    surrogates = [surrogate]
    for _ in range(new_surr_count):
        new_surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))
        torch.manual_seed(rng.integers(1 << 30))
        new_surr = train_mlp(tinf.system, new_surr_trajs)
        surrogates.append(new_surr)

    if subexp == 1:
        root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
        root_pipeline_cfg["_model:n_hidden_layers"] = "3"
        root_pipeline_cfg["_model:hidden_size_1"] = 69
        root_pipeline_cfg["_model:hidden_size_2"] = 256
        root_pipeline_cfg["_model:hidden_size_3"] = 256
        root_pipeline_cfg["_model:lr_log10"] = -3.323534
        root_pipeline_cfg["_model:nonlintype"] = "tanh"
    elif subexp == 2:
        root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
        root_pipeline_cfg["_model:n_hidden_layers"] = "3"
        root_pipeline_cfg["_model:hidden_size_1"] = 128
        root_pipeline_cfg["_model:hidden_size_2"] = 128
        root_pipeline_cfg["_model:hidden_size_3"] = 128
    else:
        raise ValueError("Unrecognized sub experiment.")

    @memory.cache
    def train_model(sysid_trajs):
        model = ampc.make_model(tinf.system, pipeline.Model, 
                pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=50,
                use_cuda=False) #C
        torch.manual_seed(eval_seed)
        model.train(sysid_trajs)
        return model.get_parameters()
    model = ampc.make_model(tinf.system, pipeline.Model, 
            pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=5,
            use_cuda=False)
    model_params = train_model(sysid_trajs)
    model.set_parameters(model_params)
    cs = pipeline.get_configuration_space_fixed_model()

    @memory.cache
    def eval_cfg(cfg_vector, surr_idx=-1):
        torch.manual_seed(eval_seed)
        cfg = Configuration(cs, vector=cfg_vector)
        #pipeline_cfg = pipeline.set_tt_cfg(root_pipeline_cfg, 0, cfg)
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model=model)
        if surr_idx == -1:
            truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
            score = tinf.perf_metric(truedyn_traj)
        else:
            surr_traj = runsim(tinf, 200, surrogates[surr_idx], controller)
            score = tinf.perf_metric(surr_traj)
        return score

    #cs = pipeline.task_transformers[0].get_configuration_space(tinf.system)
    cfgs = get_cfgs(cs)
    true_scores = []
    surr_scores = []
    for cfg in cfgs:
        true_scores.append(eval_cfg(cfg.get_array()))
        surr_scores.append([eval_cfg(cfg.get_array(), surr_idx=i) 
            for i in range(new_surr_count+1)])
    print(f"{true_scores=}")
    print(f"{surr_scores=}")

if __name__ == "__main__":
    main()
