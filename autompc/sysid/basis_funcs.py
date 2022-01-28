# Created by William Edwards (wre2@illinois.edu), 2021-02-07

import numpy as np
import inspect
from pdb import set_trace
from collections import namedtuple

BasisFunction = namedtuple("BasisFunction", ["n_args", "func", "grad_func", "name_func"])

class IdentityBasisFunction:
    n_args = 1

    def func(self):
        return 1

    def grad_func(self):
        return [0]

    def name_func(self):
        return ""

def get_constant_basis_func():
    return BasisFunction(n_args=0,
            func =      lambda : 1,
            grad_func = lambda : [0],
            name_func = lambda : "")

def get_identity_basis_func():
    return BasisFunction(n_args=1,
            func =      lambda x : x,
            grad_func = lambda x : [1],
            name_func = lambda x : x)

def get_poly_basis_func(degree):
    return BasisFunction(n_args=1,
            func =      lambda x : x**degree,
            grad_func = lambda x : [x**(degree-1)],
            name_func = lambda x : "{}**{}".format(x,degree))

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
        arg_str = ", ".join(["x{}".format(i) for i in range(n_args)])
        def func_(*args, tr):
            val = 1.0
            for arg, exp in zip(args, tr):
                val *= arg**exp
            return val

        if n_args == 1:
            func = lambda x0, *, tr=trimmed_exp: func_(x0, tr=tr)
        elif n_args == 2:
            func = lambda x0, x1, *, tr=trimmed_exp: func_(x0, x1, tr=tr)
        elif n_args == 3:
            func = lambda x0, x1, x2, *, tr=trimmed_exp: func_(x0, x1, x2, tr=tr)
        elif n_args == 4:
            func = lambda x0, x1, x2, x3, *, tr=trimmed_exp: func_(x0, x1, x2, x3, tr=tr)
        elif n_args == 5:
            func = lambda x0, x1, x2, x3, x4, *, tr=trimmed_exp: func_(x0, x1, x2, \
                    x3, x4, tr=tr)
        elif n_args == 6:
            func = lambda x0, x1, x2, x3, x4, x5, *, tr=trimmed_exp: func_(x0, x1, \
                    x2, x3, x4, x5, tr=tr)
        elif n_args == 7:
            func = lambda x0, x1, x2, x3, x4, x5, x6, *, tr=trimmed_exp: func_(x0, x1, \
                    x2, x3, x4, x5, x6, tr=tr)
        elif n_args == 8:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7, *,tr=trimmed_exp: func_(x0, x1, \
                    x2, x3, x4, x5, x6, x7, tr=tr)
        elif n_args == 9:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7, x8,*,tr=trimmed_exp: func_(x0, \
                    x1, x2, x3, x4, x5, x6, x7, x8, tr=tr)
        elif n_args == 10:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, *, tr=trimmed_exp: \
                    func_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, tr=tr)
        else:
            raise ValueError("n_args > 10")

        def grad_func(*args, trimmed_exp=trimmed_exp):
            grads = []
            for i in range(len(args)):
                val = 1.0
                for j, (arg, exp) in enumerate(zip(args, trimmed_exp)):
                    if i != j:
                        val *= arg**exp
                    else:
                        val *= exp * arg**(exp-1)
                grads.append(val)
            return np.array(grads)
        def name_func(*args, trimmed_exp=trimmed_exp):
            name = ""
            for arg, exp in zip(args, trimmed_exp):
                name += "{}^{} ".format(arg, exp)
            return name
        bfuncs.append(BasisFunction(n_args = n_args, func = func, 
            grad_func=grad_func, name_func=name_func))
    return bfuncs

def get_trig_basis_funcs(freq):
    sin_bfunc = BasisFunction(n_args=1,
            func      = lambda x : np.sin(freq * x),
            grad_func = lambda x : [freq * np.cos(freq * x)],
            name_func = lambda x : "sin({} {})".format(freq, x))
    cos_bfunc = BasisFunction(n_args=1,
            func      = lambda x : np.cos(freq * x),
            grad_func = lambda x : [-freq * np.sin(freq * x)],
            name_func = lambda x : "cos({} {})".format(freq, x))
    return [sin_bfunc, cos_bfunc]

def get_trig_interaction_terms(freq):
    sin_bfunc = BasisFunction(n_args=2,
            func      = lambda x,y : x * np.sin(freq * y),
            grad_func = lambda x,y : [np.sin(freq * y), x * freq * np.cos(freq * y)],
            name_func = lambda x,y : "{} sin({} {})".format(x, freq, y))
    sin_bfunc2 = BasisFunction(n_args=2,
            func      = lambda y,x : x * np.sin(freq * y),
            grad_func = lambda y,x : [x * freq * np.cos(freq * y), np.sin(freq * y)],
            name_func = lambda y,x : "{} sin({} {})".format(x, freq, y))
    cos_bfunc = BasisFunction(n_args=2,
            func      = lambda x,y : x * np.cos(freq * y),
            grad_func = lambda x,y : [np.cos(freq * y), x * -freq * np.sin(freq * y)],
            name_func = lambda x,y : "{} cos({} {})".format(x, freq, y))
    cos_bfunc2 = BasisFunction(n_args=2,
            func      = lambda y,x : x * np.cos(freq * y),
            grad_func = lambda y,x : [x * -freq * np.sin(freq * y), np.cos(freq * y)],
            name_func = lambda y,x : "{} cos({} {})".format(x, freq, y))
    return sin_bfunc, sin_bfunc2, cos_bfunc, cos_bfunc2

