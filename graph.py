import matplotlib.pyplot as plt
import pickle
import numpy as np

infile = open('dadIter0.data','rb')
inputData = pickle.load(infile)
x = inputData[7]
testTraj = inputData[9]

xAxisTimesteps = [0]
for t in range(1, testTraj.obs.shape[0]):
    xAxisTimesteps.append(t)


plt.plot(xAxisTimesteps, testTraj.obs[:,3].tolist(), label = "X Observation")
plt.plot(xAxisTimesteps, x.tolist(), label = "X Prediction: ")

plt.legend()
plt.savefig('trajXIndividual.png', dpi=600, bbox_inches='tight')
plt.clf()