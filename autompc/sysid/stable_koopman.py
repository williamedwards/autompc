__author__ = "Giorgos Mamakoukas"
__copyright__ = "Copyright (C) 2004 Giorgos Mamakoukas"


from pdb import set_trace

import numpy as np
from scipy.linalg import polar, pinv2, solve_discrete_lyapunov, sqrtm
import math
from scipy import io
import time


def projectPSD(Q, epsilon = 0, delta = math.inf):
    Q = (Q+Q.T)/2
    [e, V] = np.linalg.eig(Q)
    # print(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) ))
    Q_PSD = V.dot(np.diag( np.minimum( delta, np.maximum(e, epsilon) ) )).dot(V.T)
    return Q_PSD

def gradients(Xs, Xu, Y, S, U, B, Bcon):
    Sinv = np.linalg.inv(S)
    R = Sinv.dot(U).dot(B).dot(S)
    # R = np.linalg.multi_dot([Sinv, U, B, S, X])
    Error = Y - Bcon.dot(Xu)- R.dot(Xs)
    e = np.linalg.norm(Error, 'fro')

    if len(locals()) >= 2:
        temp1 = Sinv.T .dot(-Error).dot(Xs.T)
        S_grad = -temp1.dot(R.T) + B.T.dot(U.T).dot(temp1)
        U_grad = temp1.dot(S.T).dot(B.T)
        B_grad = - U.T.dot(-temp1).dot(S.T)
        Bcon_grad = - Error.dot(Xu.T)
        return e, S_grad, U_grad, B_grad, Bcon_grad
    else:
        return e

def checkdstable(A):
    n = len(A)
    P = solve_discrete_lyapunov(A.T, np.identity(n))
    S = sqrtm(P)
    invS = np.linalg.inv(S)
    UB = S.dot(A).dot(invS)
    [U,B] = polar(UB)
    B = projectPSD(B,0,1)
    return P,S,U,B

def stabilize_discrete(Xs, Xu, Y, S = None, U = None, B = None, Bcon = None, max_iter=30, time_budget=None):
    n = len(Xs) # number of Koopman basis functions
#     na2 = np.linalg.norm(Y, 'fro')**2
    na2 = np.linalg.norm(Y, 'fro')
    X = np.vstack((Xs, Xu))
    Nx = np.ma.size(Xs,0) # number of rows
    Nu = np.ma.size(Xu,0)

    print(S)
    if S is None:
        # Initialization of S, U, and B
        S = np.identity(n)
        temp = Y.dot(pinv2(X))
        [U, B] = polar(temp[:Nx,:Nx])
        B = projectPSD(B, 0, 1)
        Bcon = temp[:Nx, Nx:]

    # parameters
    alpha0 = 0.5 # parameter of FGM
    lsparam = 1.5 # parameter; has to be larger than 1 for convergence
    lsitermax = 20
    gradient = 0 # 1 for standard Gradient Descent; 0 for FGM
    print(S)
    if np.linalg.cond(S) > 1e12 :
        print(" Initial S is ill-conditioned")

    # initial step length: 1/L
    eS,_ = np.linalg.eig(S)
    L = (np.max(eS)/ np.min(eS))**2

    # Initialization
    error,_,_,_,_ = gradients(Xs,Xu,Y,S,U,B,Bcon)
    print("Error is ", error)
    step = 1/L
    i = 1
    alpha0 = 0.5
    alpha = alpha0
    Ys = S
    Yu = U
    Yb = B
    Yb_con = Bcon
    restarti = 1

    t0 = time.time()
    while i < max_iter:
        # compute gradient

        _, gS, gU, gB, gB_con = gradients(Xs,Xu, Y,S,U,B, Bcon)
        error_next = math.inf
        inner_iter = 1
        step = step * 2

        # print("This is error", error, " at iteration: ", i)
#         print("error: ", error)
        # Line Search
        while ( (error_next > error) and (  ((i == 1) and (inner_iter <= 100)) or (inner_iter <= lsitermax) ) ):
            Sn = Ys - gS*step
            Un = Yu - gU*step
            Bn = Yb - gB*step
            Bn_con = Yb_con - gB_con * step

            # Project onto feasible set
            Sn = projectPSD(Sn, 1e-15)
            Un,_ = polar(Un)
            Bn = projectPSD(Bn, 0, 1)
            # print("Projected")
            # print(Sn)
            error_next,_,_,_,_ = gradients(Xs,Xu, Y, Sn, Un, Bn, Bn_con)
            step = step / lsparam
            inner_iter = inner_iter + 1
            # print(inner_iter)
        if (i == 1):
            inner_iter0 = inner_iter

        # Conjugate with FGM weights, if cost decreased; else, restart FGM
        alpha_next = (math.sqrt(alpha**4 + 4*alpha**2) - alpha**2 )/2
        beta = alpha * (1 - alpha) / (alpha**2 + alpha_next)

        if (inner_iter >= lsitermax + 1): # line search failed
            if restarti == 1:
            # Restart FGM if not a descent direction
                restarti = 0
                alpha_next = alpha0
                Ys = S
                Yu = U
                Yb = B
                Yb_con = Bcon
                error_next = error
                print(" No descent: Restart FGM")

                # Reinitialize step length
                eS,_ = np.linalg.eig(S)
                L = (np.max(eS)/ np.min(eS))**2
                # Use information from the first step: how many steps to decrease
                step = 1/L/lsparam**inner_iter0
            elif (restarti == 0): # no previous restart/descent direction
                error_next = error
                break
        else:
            restarti = 1
            if (gradient == 1):
                beta = 0
            Ys = Sn + beta * (Sn - S)
            Yu = Un + beta * (Un - U)
            Yb = Bn + beta * (Bn - B)
            Yb_con = Bn_con + beta * (Bn_con - Bcon)
            # Keep new iterates in memory
            S = Sn
            U = Un
            B = Bn
            Bcon = Bn_con
        i = i + 1
        error = error_next
        alpha = alpha_next

        # Check if error is small (1e-6 relative error)
        if (error < 1e-12*na2):
            print("The algorithm converged")
            break
        if time_budget is not None and time.time()-t0 > time_budget:
            print("Ran out of time")
            break
    Kd = np.linalg.inv(S).dot(U).dot(B).dot(S)
    return Kd, S, U, B, Bcon, error
