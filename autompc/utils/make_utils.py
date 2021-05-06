# Created by William Edwards (wre2@illinois.edu)

def make_model(system, model, configuration, **kwargs):
    return model(system, **configuration.get_dictionary(), **kwargs)

def make_transformer(system, transformer, configuration):
    return transformer(system, **configuration.get_dictionary())

def make_controller(system, task, model, controller, configuration, **kwargs):
    return controller(system, task, model, **configuration.get_dictionary(),
            **kwargs)
