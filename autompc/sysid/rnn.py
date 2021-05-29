# Created by William Edwards (wre2@illinois.edu)

import numpy as np
import numpy.linalg as la

import torch

from ..model import Model, Hyper

class Net(torch.nn.Module):
    def __init__(self, 
            state_size,
            control_size,
            h_1 = 32,
            h_2 = 32):
        super(RnnNetwork, self).__init__()

        # TODO Initialize network layers parameters

    def forward(self, x, u, latent=None):
        #TODO Layers computation
        return xnew, latent

    def fit(X, y, batch_size, nb_epochs):
        optim = torch.optim.Adam(self.parameters())
        for i in nb_epochs:
            # Compute loss
            loss.backwards()
            opt.step()
            opt.zero_grad()


class ARX(Model):
    def __init__(self):
        # Initialize hyperparameters and parameters to default values
        pass

    def _get_training_arrays(self, trajs):
        #TODO
        pass


    def train(self, trajs):
        X, y = self._get_training_arrays(self, trajs)

        self.net = Net(trajs[0][0].shape[0], trajs[0][1].shape[0],
                h_1 = self.h_1, h_2 = self.h_2)
        self.net.fit(X, y, self.batch_size, self.nb_epochs)


    def __call__(self, xs, us, latent=None, ret_grad=False):
        if latent:
            xnew, latentnew = self.net.forward(xs[-1], us[-1], latent=latent)
        else:
            for x, u in zip(xs, us):
                xcurr, latent = self.net.forward(x, u, latent=latent)
            xnew, latentnew = xcurr, latent
        if ret_grad:
            #TODO compute xgrad, ugrad
            return xnew, latent, (xgrad, ugrad)
        else:
            return xnew, latent


    def get_hyper_options(self):
        return {"h_1" : (Hyper.int_range, (32, 256)),
                "h_2" : (Hyper.int_range, (32, 256)),
                "batch_size" : (Hyper.int_range(10, 100)),
                "nb_epochs" : (Hyper.int_range(10, 1000))}

    def get_hypers(self):
        return {"h_1" : self.h_1,
                "h_2" : self.h_2,
                "batch_size" : self.batch_size,
                "nb_epochs" : self.nb_epochs}

    def set_hypers(self, hypers):
        if "h_1" in hypers:
            self.h_1 = hypers["h_1"]
        if "h_2" in hypers:
            self.h_2 = hypers["h_2"]
        if "batch_size" in hypers:
            self.batch_size = hypers["batch_size"]
        if "nb_epochs" in hypers:
            self.nb_epochs = hypers["nb_epochs"]

    def get_parameters(self):
        return {"weights" : copy.deepcopy(self.net.state_dict())}

    def set_parameters(self, params):
        self.net.load_state_dict(params["weights"])


