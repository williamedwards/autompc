# Created by William Edwards (wre2@illinois.edu)

def make_model(system, model, configuration):
    return model(system, **configuration.get_dictionary())

def make_controller(system, task, model, controller, configuration):
    return controller(system, task, model, **configuration.get_dictionary())
