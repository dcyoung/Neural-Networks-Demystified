""" test.py """
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize


class Neural_Network(object):
    """ Docstring """

    def __init__(self):
        # Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights (Parameters)
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        """ Propagate inputs through network """
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        """ Apply a sigmoid activation function to scalar, vector or matrix input z """
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        """ Derivative of sigmoid function """
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    def costFunction(self, X, y):
        """ Compute cost for given X,y use weigfhts already stored in class"""
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        """ Compute derivative with respect to W1 and W2 """
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper methods for interacting with other methods/classes
    def getParams(self):
        """ Get W1 and W2 unrolled inot vector """
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        """ Set W1 and W2 using single parameter vector """
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end],
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(
            params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        """ Compute the gradients """
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def test(self):
        """ Docstring """
        testValues = np.arange(-5, 5, 0.01)
        plt.plot(testValues, self.sigmoid(testValues), linewidth=2)
        plt.plot(testValues, self.sigmoidPrime(testValues), linewidth=2)
        plt.grid(1)
        plt.legend(['sigmoid', 'sigmoidPrime'])
        plt.show()

    def test2(self):
        """ Docstring """
        return None


class trainer(object):
    def __init__(self, N):
        # make local reference to Neural Network:
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        """ Docstring """
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callbackF(self, params):
        """ Helps track the cost function value as we train the network """
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        """ Docstring """
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(
            X, y), options=options, callback=self.callbackF)

        # replace the original random params with the trained parameters
        self.N.setParams(_res.x)
        self.optimizationResults = _res


def computeNumericalGradient(N, X, y):
    """ Docstring """
    paramsInitial = N.getParams()
    # vector of numerical gradients for each parameter
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # set perturbation vector
        perturb[p] = e

        # add perturbation to parameter value and calc loss
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        # subtract perturbation from parameter value and calc loss
        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute slope between the those two perturbed values (compute numerical gradient)
        numgrad[p] = (loss2 - loss1) / (2 * e)

        # Return the value we changed back to zero:
        perturb[p] = 0

    # Return params to original value:
    N.setParams(paramsInitial)

    return numgrad


# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100  # Max test score is 100

# NN = Neural_Network()
# yHat = NN.forward(X)
# print yHat


# NN = Neural_Network()
# NN.test()

# Numerical gradient checking...
NN = Neural_Network()
numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X, y)
print numgrad
print grad
print np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)


# Create a network and a trainer
NN = Neural_Network()
T = trainer(NN)
# Train the network with the training data
T.train(X, y)

# Plot the cost during training
plt.plot(T.J)
plt.grid(1)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

# Use the trained network to predict values
yHat = NN.forward(X)
print yHat
