import matplotlib.pyplot as plt
import pickle
import numpy as np

infile = open('dadIter7.data','rb')
inputData = pickle.load(infile)
x = inputData[7]
testTraj = inputData[9]

xAxisTimesteps = [0]
for t in range(1, testTraj.obs.shape[0]):
    xAxisTimesteps.append(t)


plt.plot(xAxisTimesteps, testTraj.obs[:,2].tolist(), label = "X Observation")
plt.plot(xAxisTimesteps, x.tolist(), label = "X Prediction: ")

plt.legend()
plt.savefig('trajXIndividual.png', dpi=600, bbox_inches='tight')
plt.clf()

# for i in range(8):
#     infile = open('dadIter'+ str(i) + '.data','rb')
#     inputData = pickle.load(infile)
#     print(inputData[2], "\n")

model = 0
infile = open('dadIter'+ str(model) + '.data','rb')
inputData = pickle.load(infile)
dYData = inputData[4]

start = 0

if(model > 0):
    infile = open('dadIter'+ str(model - 1) + '.data','rb')
    inputData = pickle.load(infile)
    start = inputData[4].shape[0]

plt.hist(dYData[start:,2], bins=100, alpha=0.5, label="Model " + str(model))

plt.title('Model '+ str(model) + ' dY distribution X')
plt.legend()
ax = plt.gca()  # get the current axes
ax.relim()      # make sure all the data fits
ax.autoscale()
plt.savefig('dYXDistributionIndividual.png', dpi=300, bbox_inches='tight')
plt.clf()