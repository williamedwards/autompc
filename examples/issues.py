
arx = Arx(pendulum)
predobs, _ = arx.pred(traj)
# Whoops!! Model is not trained

arx.train(trajs)

arx.set_hypers(k=8)
# Whoops!! Hyperparameters are now inconsistent with training


con = FiniteHorizonLQR(pendulum, model, task)
# Gain matrix compute at this time

con.set_hypers(horizon=12)
# Recompute gain matrix?




