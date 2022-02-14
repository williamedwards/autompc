# Created by William Edwards (wre2@illinois.edu), 2021-02-07

import numpy as np
import inspect
from pdb import set_trace
from collections import namedtuple

CoargTuple = namedtuple("CoargTuple", ["co_argcount"])

class IdentityBasisFunction:
    def __init__(self):
        self.n_args = 1
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x):
        return x

    def grad_func(self, x):
        return [1]

    def name_func(self, x):
        return x

class PolyBasisFunction:
    def __init__(self, degree):
        self.n_args = 1
        self.degree = degree
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x):
        return x**self.degree

    def grad_func(self, x):
        self.degree * x**(self.degree-1)

    def name_func(self, x):
        return "{}**{}".format(x,self.degree)

class PolynomialCrossTerm:
    def __init__(self, n_args, trimmed_exp):
        self.n_args = n_args
        self.trimmed_exp = trimmed_exp
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, *args):
        val = 1.0
        for arg, exp in zip(args, self.trimmed_exp):
            val *= arg**exp
        return val

    def grad_func(self, *args):
        grads = []
        for i in range(len(args)):
            val = 1.0
            for j, (arg, exp) in enumerate(zip(args, self.trimmed_exp)):
                if i != j:
                    val *= arg**exp
                else:
                    val *= exp * arg**(exp-1)
            grads.append(val)
        return np.array(grads)

    def name_func(self, *args):
        name = ""
        for arg, exp in zip(args, self.trimmed_exp):
            name += "{}^{} ".format(arg, exp)
        return name

def get_cross_term_basis_funcs(degree):
    bfuncs = []
    exponents = np.mgrid[tuple(slice(degree) for _ in range(degree))]
    exponents = exponents.reshape((degree, -1))
    used_exps = set()
    for exp in exponents.T:
        if sum(exp) != degree:
            continue
        trimmed_exp = tuple(e for e in exp if e > 0)
        if trimmed_exp in used_exps:
            continue
        used_exps.add(trimmed_exp)
        n_args = len(trimmed_exp)
        bfuncs.append(PolynomialCrossTerm(n_args, trimmed_exp))
    return bfuncs

class SineBasisFunction:
    def __init__(self, freq):
        self.n_args = 1
        self.freq = freq
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x):
        return np.sin(self.freq * x)

    def grad_func(self, x):
        return self.freq * np.cos(self.freq * x)

    def name_func(self, x):
        return "sin({} {})".format(self.freq, x)

class CosineBasisFunction:
    def __init__(self, freq):
        self.n_args = 1
        self.freq = freq
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x):
        return np.cos(self.freq * x)

    def grad_func(self, x):
        return -self.freq * np.sin(self.freq * x)

    def name_func(self, x):
        return "cos({} {})".format(self.freq, x)

def get_trig_basis_funcs(freq):
    return [SineBasisFunction(freq), CosineBasisFunction(freq)]

class SineInteractionTerm:
    def __init__(self, freq, swap_args=False):
        self.n_args = 2
        self.freq = freq
        self.swap_args = swap_args
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x, y):
        if self.swap_args:
            x, y = y, x
        return x * np.sin(self.freq * y)

    def grad_func(self, x, y):
        if self.swap_args:
            x, y = y, x
        grad = [np.sin(self.freq * y), x * self.freq * np.cos(self.freq * y)]
        if self.swap_args:
            grad[0], grad[1] = grad[1], grad[0]
        return grad

    def name_func(self, x, y):
        if self.swap_args:
            x, y = y, x
        return "{} sin({} {})".format(x, self.freq, y)

class CosineInteractionTerm:
    def __init__(self, freq, swap_args=False):
        self.n_args = 2
        self.freq = freq
        self.swap_args = swap_args
        self.__code__ = CoargTuple(self.n_args)

    def __call__(self, x, y):
        if self.swap_args:
            x, y = y, x
        return x * np.cos(self.freq * y)

    def grad_func(self, x, y):
        if self.swap_args:
            x, y = y, x
        grad = [np.cos(self.freq * y), x * -self.freq * np.sin(self.freq * y)]
        if self.swap_args:
            grad[0], grad[1] = grad[1], grad[0]
        return grad

    def name_func(self, x, y):
        if self.swap_args:
            x, y = y, x
        return "{} cos({} {})".format(x, self.freq, y)

def get_trig_interaction_terms(freq):
    return [SineInteractionTerm(freq), SineInteractionTerm(freq, swap_args=True),
            CosineInteractionTerm(freq), CosineInteractionTerm(freq, swap_args=True)]

