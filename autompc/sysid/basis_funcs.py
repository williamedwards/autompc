# Created by William Edwards (wre2@illinois.edu), 2021-02-07

import numpy as np
import inspect
from collections import namedtuple

BasisFunction = namedtuple("BasisFunction", ["n_args", "func", "grad_func", "name_func"])

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
    for exp in exponents:
        if sum(exp) != degree:
            continue
        trimmed_exp = tuple(e for e in exp if e > 0)
        if trimmed_exp in used_exps:
            continue
        used_exps.add(trimmed_exp)
        n_args = len(trimmed_exp)
        arg_str = ", ".join(["x{}".format(i) for i in range(n_args)])
        def func_(*args):
            val = 1.0
            for arg, exp in zip(args, trimmed_exp):
                val *= arg**exp
            return val

        if n_args == 1:
            func = lambda x0: func_(x0)
        elif n_args == 2:
            func = lambda x0, x1: func_(x0, x1)
        elif n_args == 3:
            func = lambda x0, x1, x2: func_(x0, x1, x2)
        elif n_args == 4:
            func = lambda x0, x1, x2, x3: func_(x0, x1, x2, x3)
        elif n_args == 5:
            func = lambda x0, x1, x2, x3, x4: func_(x0, x1, x2, x3, x4)
        elif n_args == 6:
            func = lambda x0, x1, x2, x3, x4, x5: func_(x0, x1, \
                    x2, x3, x4, x5)
        elif n_args == 7:
            func = lambda x0, x1, x2, x3, x4, x5, x6: func_(x0, x1, \
                    x2, x3, x4, x5, x6)
        elif n_args == 8:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7: func_(x0, x1, \
                    x2, x3, x4, x5, x6, x7)
        elif n_args == 9:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7, x8: func_(x0, \
                    x1, x2, x3, x4, x5, x6, x7, x8)
        elif n_args == 10:
            func = lambda x0, x1, x2, x3, x4, x5, x6, x7, x8, x9: \
                    func_(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
        else:
            raise ValueError("n_args > 10")

        def grad_func(*args):
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
        def name_func(*args):
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
