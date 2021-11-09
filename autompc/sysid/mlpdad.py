"""
Just doing system id using multi-layer perceptron.
The code is similar to GP / RNN.
The configuration space has to be carefully considered
"""
import itertools
import numpy as np
from tqdm import tqdm
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

import matplotlib.pyplot as plt
import pickle

from pdb import set_trace

from .model import Model, ModelFactory
from autompc.sysid import model

def transform_input(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] - xu_means[i]) / xu_std[i])
    return np.vstack(XUt).T

def transform_output(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] * xu_std[i]) + xu_means[i])
    return np.vstack(XUt).T

class ForwardNet(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes, nonlintype):
        """Specify the feedforward neuro network size and nonlinearity"""
        assert len(hidden_sizes) > 0
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.ModuleDict() # a collection that will hold your layers
        last_n = n_in
        for i, size in enumerate(hidden_sizes):
            self.layers['layer%d' % i] = torch.nn.Linear(last_n, size)
            last_n = size
        # the final one
        self.output_layer = torch.nn.Linear(last_n, n_out)
        if nonlintype == 'relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlintype == 'selu':
            self.nonlin = torch.nn.SELU()
        elif nonlintype == 'tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlintype == 'sigmoid':
            self.nonlin = torch.nn.Sigmoid()
        else:
            raise NotImplementedError("Currently supported nonlinearity: relu, tanh, sigmoid")

    def forward(self, x):
        for i, lyr in enumerate(self.layers):
            y = self.layers[lyr](x)
            x = self.nonlin(y)
        return self.output_layer(x)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


class MLPDADFactory(ModelFactory):
    """
    The multi-layer perceptron (MLP) model uses a feed-forward neural network
    architecutret to predict the system dynamics. The network size, activation
    function, and learning rate are tunable hyperparameters.

    Parameters

    - *n_batch* (Type: int, Default: 64): Training batch size of the neural net.
    - *n_train_iters* (Type: int, Default: 50): Number of training epochs

    Hyperparameters:

    - *n_hidden_layers* (Type: str, Choices: ["1", "2", "3", "4"], Default: "2"):
      The number of hidden layers in the network
    - *hidden_size_1* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 1.
    - *hidden_size_2* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 2. 
      (Conditioned on n_hidden_layers >=2).
    - *hidden_size_3* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 3. 
      (Conditioned on n_hidden_layers >=3).
    - *hidden_size_4* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 4. 
      (Conditioned on n_hidden_layers >=4).
    - *nonlintype* (Type: str, choices: ["relu", "tanh", "sigmoid", "selu"], Default: "relu):
      Type of activation function.
    - *lr* (Type: float, Low: 1e-5, High: 1, Default: 1e-3): Adam learning rate for the network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Model = MLPDAD
        self.name = "MLPDAD"

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        nonlintype = CSH.CategoricalHyperparameter("nonlintype", 
                choices=["relu", "tanh", "sigmoid", "selu"],
                default_value="relu")
                #choices=["relu"])
        n_hidden_layers = CSH.CategoricalHyperparameter("n_hidden_layers",
                choices=["1", "2", "3", "4"], default_value="2")
        hidden_size_1 = CSH.UniformIntegerHyperparameter("hidden_size_1",
                lower = 16, upper = 256, default_value=128)
        hidden_size_2 = CSH.UniformIntegerHyperparameter("hidden_size_2",
                lower = 16, upper = 256, default_value=128)
        hidden_size_3 = CSH.UniformIntegerHyperparameter("hidden_size_3",
                lower = 16, upper = 256, default_value=128)
        hidden_size_4 = CSH.UniformIntegerHyperparameter("hidden_size_4",
                lower = 16, upper = 256, default_value=128)
        hidden_cond_2 = CSC.InCondition(child=hidden_size_2, parent=n_hidden_layers,
                values=["2","3","4"])
        hidden_cond_3 = CSC.InCondition(child=hidden_size_3, parent=n_hidden_layers,
                values=["3","4"])
        hidden_cond_4 = CSC.InCondition(child=hidden_size_4, parent=n_hidden_layers,
                values=["4"])
        lr = CSH.UniformFloatHyperparameter("lr",
                lower = 1e-5, upper = 1, default_value=1e-3, log=True)
        cs.add_hyperparameters([nonlintype, n_hidden_layers, hidden_size_1,
            hidden_size_2, hidden_size_3, hidden_size_4,
            lr])
        cs.add_conditions([hidden_cond_2, hidden_cond_3, hidden_cond_4])
        return cs

class MLPDAD(Model):
    def __init__(self, system, n_hidden_layers=3, hidden_size=128, 
            nonlintype='relu', n_train_iters=50, n_batch=64, lr=1e-3,
            hidden_size_1=None, hidden_size_2=None, hidden_size_3=None,
            hidden_size_4=None, seed=200,
            use_cuda=True,
            n_dad_iters=2): # TODO: Find good default
        Model.__init__(self, system)
        nx, nu = system.obs_dim, system.ctrl_dim
        n_hidden_layers = int(n_hidden_layers)
        hidden_sizes = [hidden_size] * n_hidden_layers
        #print(f"{torch.cuda.is_available()=}")
        if use_cuda and torch.cuda.is_available():
            print("MLPDAD Using Cuda")
        else:
            if use_cuda:
                print("MLPDAD Not Using Cuda because torch.cuda is not available")
            else:
                print("MLPDAD Not Using Cuda")
        for i, size in enumerate([hidden_size_1, hidden_size_2, hidden_size_3,
                hidden_size_4]):
            if size is not None:
                hidden_sizes[i] = size
        print("hidden_sizes=", hidden_sizes)
        torch.manual_seed(seed)
        self.net = ForwardNet(nx + nu, nx, hidden_sizes, nonlintype)
        self._train_data = (n_train_iters, n_batch, lr, n_dad_iters)
        self._device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        self.net = self.net.double().to(self._device)
        self.torchseed = seed

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def trainMLP(self, XU, dY): #Set self.net to the net you want to train before
        torch.manual_seed(self.torchseed)
        n_iter, n_batch, lr, n_dad_iter = self._train_data

        print("Training MLP with \nXU:", str(XU.shape),"\n", XU, "\ndY:", str(dY.shape),"\n", dY)

        self.xu_means = np.mean(XU, axis=0)
        self.xu_std = np.std(XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, XU)

        self.dy_means = np.mean(dY, axis=0)
        self.dy_std = np.std(dY, axis=0)
        dYt = transform_input(self.dy_means, self.dy_std, dY)

        if(XU.shape[0] != 39800):
            goodIndices = np.arange(0, 39801).tolist()

            for i in range(39800, XUt.shape[0]):
                if(np.linalg.norm(XUt[i,:-1]) < .2):
                    goodIndices.append(i)
                    # XUt = np.delete(XUt, i, 0)
                    # dYt = np.delete(dYt, i, 0)

            XUt = XUt[goodIndices, :]
            dYt = dYt[goodIndices, :]

            XUtemp = transform_output(self.xu_means, self.xu_std, XUt)
            dYtemp = transform_output(self.dy_means, self.dy_std, dYt)

            self.xu_means = np.mean(XUtemp, axis=0)
            self.xu_std = np.std(XUtemp, axis=0)
            XUt = transform_input(self.xu_means, self.xu_std, XUtemp)

            self.dy_means = np.mean(dYtemp, axis=0)
            self.dy_std = np.std(dYtemp, axis=0)
            dYt = transform_input(self.dy_means, self.dy_std, dYtemp)


        feedX = XUt
        predY = dYt
        dataset = SimpleDataset(feedX, predY)
        dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        # now I can perform training... using torch default dataset holder
        self.net.train()
        for param in self.net.parameters():
            param.requires_grad_(True)
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossfun = torch.nn.SmoothL1Loss()

        for i in tqdm(range(n_iter), file=sys.stdout):
            cum_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                x = x.to(self._device)
                predy = self.net(x)
                loss = lossfun(predy, y.to(self._device))
                loss.backward()
                cum_loss += loss.item()
                optim.step()
            print("Loss: ", cum_loss)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

        return (self.xu_means, self.xu_std, self.dy_means, self.dy_std)

    def train(self, trajs, silent=False, seed=200):
        torch.manual_seed(seed)
        n_iter, n_batch, lr, n_dad_iter = self._train_data

        print("Initial self.net id: \t", id(self.net))
        originalNet = copy.deepcopy(self.net)
        print("Original Net id: \t", id(originalNet))

        # Initial Training of Model
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together

        print("\nTraining Initial MLP: ", end="\n")
        normalizationParams = self.trainMLP(XU, dY) # self.net gets trained and updates normalization Params internally
        print("Init train self.net id: \t", id(self.net))

        # trainedModels = [copy.deepcopy(self.net)]
        trainedModels = [self.net]
        print("self.net in list id: \t", id(trainedModels[0]))

        lossfun = torch.nn.SmoothL1Loss()
        lossfun = torch.nn.MSELoss()
        modelsLoss = [self.evaluateAccuracy(trajs, lossfun)]
        print("Initial Model error: ", modelsLoss[0])

        modelsNormParams = [normalizationParams]
        print("Initial Model norm params: ", modelsNormParams[0])


        # Debug for comparing trajectories on DaD iteration
        XData = [copy.deepcopy(X)]
        dYData = [copy.deepcopy(dY)]

        # Evaluate for first model
        testTrajectory = trajs[0]
        predictTraj = self.generatePredictedTrajectoryObservations(testTrajectory[0].obs, testTrajectory.ctrls, maxTimestep=testTrajectory.obs.shape[0] - 1)

        thetaData = [predictTraj[:,0]]
        omegaData = [predictTraj[:,1]]
        xData = [predictTraj[:,2]]
        dxData = [predictTraj[:,3]]
        

        print("\nTraining MLP with DAD: ", end="\n")
        for n in range(n_dad_iter):
            print("DaD Iteration", n + 1, "of", n_dad_iter, end="\n")
            # Reset dataset to initial state
            # X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
            # dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
            # U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs]) 
            print("Generating Predicted Trajectories: ", end="\n")
            for traj in tqdm(trajs, file=sys.stdout):
                # Generate predictions into a trajectory of 0 through T - 1
                predictedTrajectory = self.generatePredictedTrajectoryObservations(traj[0].obs, traj.ctrls, maxTimestep=traj.obs.shape[0] - 2)
                #print("Observed traj: \n", traj.obs)
                #print("\nObserved prediction: (We want to use [1, T-1])\n", predictedTrajectory)

                # Adding feedX values
                # X = np.concatenate((X, predictedTrajectory[1:])) #Exclude xhat 0 as it is an observed value
                #print("\nFull X is now: \n", X)

                # Predicted traj is {0,1,...,T-1}                    
                xi = traj.obs[2:] # From t = 2 to t = T
                xhat = predictedTrajectory[1:] # From t = 1 to t = T - 1
                newDY = xi - xhat
                #print("Xi values from T [2,T]: \n", xi)
                #print("xhat values from T [1,T-1] \n", xhat)
                #print("dY to be appended: \n", newDY)

                newU = traj.ctrls[1:-1]

                # tempX = np.empty(shape=[0, newDY.shape[1]])
                # tempU = np.empty(shape=[0, traj.ctrls.shape[1]])
                # tempNewDY = np.empty(shape=[0, newDY.shape[1]])
                tempX = np.array([np.zeros(newDY.shape[1])])
                tempU = np.array([np.zeros(traj.ctrls.shape[1])])
                tempNewDY = np.array([np.zeros(newDY.shape[1])])

                keepIndices = []

                for i in range(newDY.shape[0]):
                    if(np.linalg.norm(newDY[i]) >= 0):
                    #if(np.max(newDY[i]) <= .1):    
                        #print(newDY[i], " Norm: ", np.linalg.norm(newDY[i]), " Index: ", i)
                        tempX = np.vstack((tempX, xhat[i]))
                        tempNewDY = np.vstack((tempNewDY,newDY[i]))
                        tempU = np.vstack((tempU, newU[i]))
                        keepIndices.append(i)

                futureX = np.delete(tempX, 0, 0)
                newDY = np.delete(tempNewDY, 0, 0)
                newU = np.delete(tempU, 0, 0)


                X = np.concatenate((X, futureX)) #Exclude xhat 0 as it is an observed value
                
                #print("old dY: \n", dY)
                dY = np.concatenate((dY, newDY))
                #print("dY with newDY: \n", dY)

                #print("old U: \n", U)
                #U = np.concatenate((U, traj.ctrls[1:-1]))
                U = np.concatenate((U, newU))
                #print("new U: \n", U)
    
            XU = np.concatenate((X, U), axis=1) # stack X and U together as X | U
            #print("Combined XU: \n", XU)


            
            

            # train Nth model on untrained model
            print("Training MLP: ", end="\n")
            self.net = copy.deepcopy(originalNet) # Reset self.net for training
            print("before train self.net id model #", n + 2, ":\t", id(self.net))
            normalizationParams = self.trainMLP(XU, dY)
            print("after train self.net id model #", n + 2, ":\t", id(self.net))

            

            # TODO: Add evaluation for cumulative loss based on hold out dataset

            # Debug for comparing trajectories on DaD iteration
            #trainedModels.append(copy.deepcopy(self.net))
            trainedModels.append(self.net)
            modelsNormParams.append(normalizationParams)
            predictionError = self.evaluateAccuracy(trajs, lossfun)
            modelsLoss.append(predictionError)


            predictTraj = self.generatePredictedTrajectoryObservations(testTrajectory[0].obs, testTrajectory.ctrls, maxTimestep=testTrajectory.obs.shape[0] - 1)

            thetaData.append(predictTraj[:,0])
            omegaData.append(predictTraj[:,1])
            xData.append(predictTraj[:,2])
            dxData.append(predictTraj[:,3])

            dYData.append(copy.deepcopy(dY))
            XData.append(copy.deepcopy(X))


            print("\n\t\t\t\t\t\t\t xu_means, xu_std, dy_means, dy_std")
            for i in range(0, len(trainedModels)):
                print("Model #", i, id(trainedModels[i]), "\tLoss: ", modelsLoss[i])#, "\tNorm Params: ", modelsNormParams[i])


        minError = min(modelsLoss)
        min_index = modelsLoss.index(minError)

        self.net = trainedModels[min_index]

        normalizationParams = modelsNormParams[min_index]
        self.xu_means= normalizationParams[0]
        self.xu_std= normalizationParams[1]
        self.dy_means= normalizationParams[2]
        self.dy_std= normalizationParams[3]

        xAxisTimesteps = [0]
        for t in range(1, trajs[0].obs.shape[0]):
            xAxisTimesteps.append(t)

        plt.plot(xAxisTimesteps, testTrajectory.obs[:,0].tolist(), label = "Theta Observation")
        for n in range(n_dad_iter + 1):
            plt.plot(xAxisTimesteps, thetaData[n].tolist(), label = "Theta Prediction: " + str(n))

        plt.legend()
        plt.savefig('trajTheta.png', dpi=600, bbox_inches='tight')
        plt.clf()

        plt.plot(xAxisTimesteps, testTrajectory.obs[:,1].tolist(), label = "Omega Observation")
        for n in range(n_dad_iter + 1):
            plt.plot(xAxisTimesteps, omegaData[n].tolist(), label = "Omega Prediction: " + str(n))

        plt.legend()
        plt.savefig('trajOmega.png', dpi=600, bbox_inches='tight')
        plt.clf()

        plt.plot(xAxisTimesteps, testTrajectory.obs[:,2].tolist(), label = "X Observation")
        for n in range(n_dad_iter + 1):
            plt.plot(xAxisTimesteps, xData[n].tolist(), label = "X Prediction: " + str(n))

        plt.legend()
        plt.savefig('trajX.png', dpi=600, bbox_inches='tight')
        plt.clf()

        plt.plot(xAxisTimesteps, testTrajectory.obs[:,3].tolist(), label = "dX Observation")
        for n in range(n_dad_iter + 1):
            plt.plot(xAxisTimesteps, dxData[n].tolist(), label = "dX Prediction: " + str(n))

        plt.legend()
        plt.savefig('trajdX.png', dpi=600, bbox_inches='tight')
        plt.clf()


        # Variable Distribution Graphs
        for i in range(len(dYData)):
            plt.hist(dYData[i][:,0], bins=100, alpha=0.5, label="Model " + str(i))

        plt.legend()
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()
        plt.savefig('dYThetaDistribution.png', dpi=300, bbox_inches='tight')
        plt.clf()


        for i in range(len(dYData)):
            plt.hist(dYData[i][:,1], bins=100, alpha=0.5, label="Model " + str(i))

        plt.legend()
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()
        plt.savefig('dYOmegaDistribution.png', dpi=300, bbox_inches='tight')
        plt.clf()


        for i in range(len(dYData)):
            plt.hist(dYData[i][:,2], bins=100, alpha=0.5, label="Model " + str(i))

        plt.legend()
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()
        plt.savefig('dYXDistribution.png', dpi=300, bbox_inches='tight')
        plt.clf()


        for i in range(len(dYData)):
            plt.hist(dYData[i][:,3], bins=100, alpha=0.5, label="Model " + str(i))

        plt.legend()
        ax = plt.gca()  # get the current axes
        ax.relim()      # make sure all the data fits
        ax.autoscale()
        plt.savefig('dYdXDistribution.png', dpi=300, bbox_inches='tight')
        plt.clf()

        # for n in range(n_dad_iter + 1):
        #     np.savez('dadIter_' + str(n), X=XData[n], dY=dYData[n], loss=np.array(modelsLoss))

        for n in range(n_dad_iter + 1):
            outfile = open('dadIter' + str(n) + '.data','wb')
            out = [trainedModels[n], modelsLoss[n], modelsNormParams[n], XData[n], dYData[n], thetaData[n], omegaData[n], xData[n], dxData[n], testTrajectory, trajs]
            pickle.dump(out,outfile)
            outfile.close()

        # Output Data
        # trainedModels = [self.net]

        # modelsLoss = [self.evaluateAccuracy(trajs, lossfun)]

        # modelsNormParams = [normalizationParams]

        # # Debug for comparing trajectories on DaD iteration
        # XData = [copy.deepcopy(X)]
        # dYData = [copy.deepcopy(dY)]

        # thetaData = [predictTraj[:,0]]
        # omegaData = [predictTraj[:,1]]
        # xData = [predictTraj[:,2]]
        # dxData = [predictTraj[:,3]]

        
        

    def evaluateAccuracy(self, trajss, lossfun):
        #observations = trajss.obs[:,0]
        #trajs = trajss.obs[np.where(observations >= 0, observations <= np.pi * 2)]
        trajs = []
        for traj in trajss:
            if(abs(np.max(traj.obs[:,0])) <= (np.pi * 2) and abs(np.min(traj.obs[:,0])) >= 0):
                trajs.append(traj)
            else:
                print("Bad Traj: ", np.max(traj.obs[:,0]), " ", np.min(traj.obs[:,0]))

        #trajs = trajss[50:80]

        debug = [0]
        cum_loss = 0

        badpredictCount = 0
        for traj in trajs:
            predTraj = self.generatePredictedTrajectoryObservations(traj[0].obs, traj.ctrls[:-1]) # Exclude T + 1
            #difference = predTraj[1:] - traj.obs[1:] # Exclude initial observation # For debug
            temploss = 0
            for t in range(1, traj.obs.shape[0]):
                predY = predTraj[t]
                y = traj[t].obs
                loss = lossfun(torch.from_numpy(predTraj[t]), torch.from_numpy(traj[t].obs))
                cum_loss += loss.item()
                temploss += loss.item()
                debug.append(loss.item())
                #if(loss.item() > 1000000):
                    #print(loss.item())
                    #breakpoint()
            
            if(temploss > 100000000):
                badpredictCount += 1
                
        #breakpoint()
        print(badpredictCount)
        return cum_loss/len(trajs)

    # self.net and the correct xu and dy means need to be set before running this method
    def generatePredictedTrajectoryObservations(self, initialState, ctrls, maxTimestep=-1):
        predictedTrajectory = np.array([initialState])
        # Will make predictions for all control inputs
        timesteps = ctrls.shape[0] # 20 controls will create timesteps 0 through 20 which is 21 timesteps.
        if(timesteps > maxTimestep and maxTimestep != -1):
            timesteps = maxTimestep

        for t in range(0, timesteps):
            predictedTrajectory = np.concatenate((predictedTrajectory, np.array([self.pred(predictedTrajectory[t], ctrls[t])])))
        return predictedTrajectory
                    

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy

    def pred_batch(self, state, ctrl):
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy.reshape((state.shape[0], self.state_dim))

    def pred_diff(self, state, ctrl):
        """Use code from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa .
        
        def get_batch_jacobian(net, x, to):
            # noutputs: total output dim (e.g. net(x).shape(b,1,4,4) noutputs=1*4*4
            # b: batch
            # i: in_dim
            # o: out_dim
            # ti: total input dim
            # to: total output dim
            x_batch = x.shape[0]
            x_shape = x.shape[1:]
            x = x.unsqueeze(1)  # b, 1 ,i
            x = x.repeat(1, to, *(1,)*len(x.shape[2:]))  # b * to,i  copy to o dim
            x.requires_grad_(True)
            tmp_shape = x.shape
            y = net(x.reshape(-1, *tmp_shape[2:]))  # x.shape = b*to,i y.shape = b*to,to
            y_shape = y.shape[1:]  # y.shape = b*to,to
            y = y.reshape(x_batch, to, to)  # y.shape = b,to,to
            input_val = torch.eye(to).reshape(1, to, to).repeat(x_batch, 1, 1)  # input_val.shape = b,to,to  value is (eye)
            y.backward(input_val)  # y.shape = b,to,to
            return x.grad.reshape(x_batch, *y_shape, *x_shape).data  # x.shape = b,o,i
        """
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        xin = torch.from_numpy(Xt).to(self._device)
        n_out = self.system.obs_dim
        xin = xin.repeat(n_out, 1)
        xin.requires_grad_(True)
        yout = self.net(xin)
        # compute gradient...
        yout.backward(torch.eye(n_out).to(self._device))
        jac = xin.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac / self.xu_std[None] * self.dy_std[:, np.newaxis]
        n = self.system.obs_dim
        state_jac = jac[:, :n] + np.eye(n)
        ctrl_jac = jac[:, n:]
        out = yout.detach().cpu().numpy()[:1, :]
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state+dy, state_jac, ctrl_jac

    def pred_diff_batch(self, state, ctrl):
        """Prediction, but with gradient information"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = state.shape[1]
        m = state.shape[0]
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self._device)
        TsrXt = TsrXt.repeat(obs_dim, 1, 1).permute(1,0,2).flatten(0,1)
        TsrXt.requires_grad_(True)
        predy = self.net(TsrXt)
        predy.backward(torch.eye(obs_dim).to(self._device).repeat(m,1), retain_graph=True)
        predy = predy.reshape((m, obs_dim, obs_dim))
        #predy.backward(retain_graph=True)
        jac = TsrXt.grad.cpu().data.numpy()
        jac = jac.reshape((m, obs_dim, TsrXt.shape[-1]))
        # properly scale back...
        jac = jac / np.tile(self.xu_std, (m,obs_dim,1)) * np.tile(self.dy_std, (m,1))[:,:,np.newaxis]
        # since repeat, y value is the first one...
        out = predy[:,0,:].cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out)
        n = self.system.obs_dim
        state_jacs = jac[:, :, :n] + np.tile(np.eye(n), (m,1,1))
        ctrl_jacs = jac[:, :, n:]
        return state + dy, state_jacs, ctrl_jacs


    def get_parameters(self):
        return {"net_state" : self.net.state_dict(),
                "xu_means" : self.xu_means,
                "xu_std" : self.xu_std,
                "dy_means" : self.dy_means,
                "dy_std" : self.dy_std }


    def set_parameters(self, params):
        self.xu_means = params["xu_means"]
        self.xu_std = params["xu_std"]
        self.dy_means = params["dy_means"]
        self.dy_std = params["dy_std"]
        self.net.load_state_dict(params["net_state"])
