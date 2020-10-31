# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace
import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from ..cs_utils import *
from ..utils import *

class FixedControlPipeline:
    def __init__(self, system, task, Model, Controller, task_transformers, 
            controller_kwargs=dict(), use_cuda=False):
        self.system = system
        self.task = task
        self.Model = Model
        self.Controller = Controller
        self.task_transformers = task_transformers[:]
        self.controller_kwargs = controller_kwargs
        self.use_cuda = use_cuda


    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        model_cs = self.Model.get_configuration_space(self.system)
        add_configuration_space(cs, "_model", model_cs)
        contr_cs = self.Controller.get_configuration_space(self.system, self.task,
                self.Model)
        add_configuration_space(cs, "_controller", contr_cs)
        for i, trans in enumerate(self.task_transformers):
            trans_cs = trans.get_configuration_space(self.system)
            add_configuration_space(cs, "_task_transformer_{}".format(i), trans_cs)
        return cs

    def get_configuration_space_fixed_model(self):
        cs = CS.ConfigurationSpace()
        contr_cs = self.Controller.get_configuration_space(self.system, self.task,
                self.Model)
        add_configuration_space(cs, "_controller", contr_cs)
        for i, trans in enumerate(self.task_transformers):
            trans_cs = trans.get_configuration_space(self.system)
            add_configuration_space(cs, "_task_transformer_{}".format(i), trans_cs)
        return cs

    def set_configuration_fixed_model(self, root_cfg, child_cfg):
        cfg = copy.deepcopy(root_cfg)
        transfer_subspace_configuration(child_cfg, "_controller", cfg,
                "_controller")
        for i, trans in enumerate(self.task_transformers):
            transfer_subspace_configuration(child_cfg, f"_task_transformer_{i}", 
                    cfg, f"_task_transformer_{i}")
        return cfg


    def set_model_cfg(self, pipeline_cfg, model_cfg):
        cfg = copy.deepcopy(pipeline_cfg)
        set_parent_configuration(cfg, "_model", model_cfg)
        return cfg

    def set_controller_cfg(self, pipeline_cfg, controller_cfg):
        cfg = copy.deepcopy(pipeline_cfg)
        set_parent_configuration(cfg, "_controller", controller_cfg)
        return cfg

    def set_tt_cfg(self, pipeline_cfg, tt_idx, tt_cfg):
        cfg = copy.deepcopy(pipeline_cfg)
        set_parent_configuration(cfg, f"_task_transformer_{tt_idx}", tt_cfg)
        return cfg

    def get_model_cfg(self, pipeline_cfg):
        model_cs = self.Model.get_configuration_space(self.system)
        model_cfg = model_cs.get_default_configuration()
        set_subspace_configuration(pipeline_cfg, "_model", model_cfg)
        return model_cfg

    def __call__(self, cfg, trajs, model=None):
        print("Making pipeline configuration")
        print(cfg)
        print("-----")
        # First instantiate and train the model
        print(f"{self.use_cuda=}")
        if model is None:
            model_cfg = self.get_model_cfg(cfg)
            model = make_model(self.system, self.Model, model_cfg,
                    use_cuda=self.use_cuda)
            model.train(trajs)
            print("Exit training.")

        # Next set up task
        task = self.task
        for i, Trans in enumerate(self.task_transformers):
            trans_cs = Trans.get_configuration_space(self.system)
            trans_cfg = trans_cs.get_default_configuration()
            set_subspace_configuration(cfg, "_task_transformer_{}".format(i), trans_cfg)
            trans = make_transformer(self.system, Trans, trans_cfg)
            task = trans(task, trajs)

        # Finally create the controller
        contr_cs = self.Controller.get_configuration_space(self.system, self.task,
                model)
        contr_cfg = contr_cs.get_default_configuration()
        set_subspace_configuration(cfg, "_controller", contr_cfg)
        controller = make_controller(self.system, task, model, self.Controller, 
                contr_cfg, **self.controller_kwargs)

        return controller, model

        
