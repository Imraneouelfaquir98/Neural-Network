import numpy as np
import matplotlib.pyplot as plt

class NeuralNets():
    def __init__(
        self, 
        X, 
        Y, 
        n_h            = 4, 
        num_iterations = 10000, 
        print_cost     = False,
        nbr_print = 100
    ):
        self.X = X
        self.Y = Y
        self.n_h = n_h
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.nbr_print = nbr_print

    def layer_sizes(self):
        n_x = self.X.shape[0] # size of input layer
        n_y = self.Y.shape[0] # size of output layer
        return (n_x, self.n_h, n_y)
        
    def initialize_parameters(self, n_x, n_y):
        np.random.seed(2) 
        W1 = np.random.randn(self.n_h, n_x) * 0.01
        b1 = np.zeros((self.n_h,1))
        W2 = np.random.randn(n_y, self.n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        assert (W1.shape == (self.n_h, n_x))
        assert (b1.shape == (self.n_h, 1))
        assert (W2.shape == (n_y, self.n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

    def forward_propagation(self, X, parameters):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        return A2, cache

    def compute_cost(self, A2, parameters):
        m = self.Y.shape[1]
        logprobs = 1/m*(np.multiply(np.log(A2),self.Y) + np.multiply(np.log(1 - A2),1 - self.Y))
        cost = - np.sum(logprobs) 
        cost = float(np.squeeze(cost))
        assert(isinstance(cost, float))
        return cost

    # GRADED FUNCTION: backward_propagation
    def backward_propagation(self, parameters, cache):
        m = self.X.shape[1]
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        dZ2 = A2 - self.Y
        dW2 = 1/m*np.dot(dZ2, A1.T)
        db2 = 1/m*np.sum(dZ2, axis = 1, keepdims = True)
        dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
        dW1 = 1/m*np.dot(dZ1, self.X.T)
        db1 = 1/m*np.sum(dZ1, axis = 1, keepdims = True)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        return grads

    def update_parameters(self, parameters, grads, learning_rate = 1.2):
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

    def fit(self):
        np.random.seed(3)
        n_x = self.layer_sizes()[0]
        n_y = self.layer_sizes()[2]
        
        parameters = self.initialize_parameters(n_x, n_y)
        for i in range(0, self.num_iterations):
            A2, cache = self.forward_propagation(self.X, parameters)
            cost = self.compute_cost(A2, parameters)
            grads = self.backward_propagation(parameters, cache)
            parameters = self.update_parameters(parameters, grads)
            if self.print_cost and i % self.nbr_print == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        return parameters

    def predict(self, parameters, X):
        A2, cache = self.forward_propagation(X, parameters)
        predictions = self.sigmoid(A2)
        return predictions

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    
class DeepNeuralNets():
    def __init__(
        self,
        X, 
        Y, 
        layers_dims, 
        learning_rate = 0.075, 
        num_iterations = 10000, 
        initialization = "random", 
        optimizer="gd",
        beta=0.9, 
        beta1=0.9, 
        beta2=0.999, 
        epsilon=1e-8, 
        print_cost=False,
        nbr_print = 100
    ):
        self.X = X
        self.Y = Y
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.initialization = initialization
        self.optimizer = optimizer
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.print_cost = print_cost
        self.nbr_print = nbr_print
    
    def sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache
    
    def relu(self, Z):
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache
    
    def relu_backward(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ
    
    def sigmoid_backward(self, dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def initialize_parameters_random(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.layers_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1]) * 0.1
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
            assert(parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (self.layers_dims[l], 1))
        return parameters

    def initialize_parameters_zeros(self):
        parameters = {}
        L = len(self.layers_dims)        
        for l in range(1, L):
            parameters['W' + str(l)] = np.zeros((self.layers_dims[l], self.layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (self.layers_dims[l], 1))
        return parameters

    def initialize_parameters_xavier(self):
        parameters = {}
        L = len(self.layers_dims) - 1 
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1])*np.sqrt(1/self.layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l],1))
        assert(parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (self.layers_dims[l], 1))
        return parameters
    
    def initialize_parameters_he(self):
        parameters = {}
        L = len(self.layers_dims) - 1 
        for l in range(1, L + 1):
            parameters['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l-1])*np.sqrt(2/self.layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros((self.layers_dims[l],1))
        assert(parameters['W' + str(l)].shape == (self.layers_dims[l], self.layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (self.layers_dims[l], 1))
        return parameters
    
    def linear_forward(self, A, W, b):
        Z = np.dot(W,A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache
    
    def L_model_forward(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
        caches.append(cache)
        assert(AL.shape == (1,X.shape[1]))
        return AL, caches
    
    def compute_cost(self, AL):
        m = self.Y.shape[1]
        cost = -1/m*np.sum(np.multiply(np.log(AL),self.Y)+np.multiply(np.log(1-AL),1-self.Y))
        cost = np.squeeze(cost)    
        assert(cost.shape == ())
        return cost
    
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1/m * np.dot(dZ,A_prev.T)
        db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T,dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        self.Y = self.Y.reshape(AL.shape)
        dAL = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads
    
    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - self.learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - self.learning_rate * grads["db" + str(l+1)]
        return parameters
    
    def update_parameters_with_gd(self, parameters, grads):
        L = len(parameters) // 2 
        for l in range(L):
            parameters["W" + str(l+1)] -= self.learning_rate*grads['dW' + str(l+1)]
            parameters["b" + str(l+1)] -= self.learning_rate*grads['db' + str(l+1)]
        return parameters

    def initialize_velocity(self, parameters):
        L = len(parameters) // 2 
        v = {}
        for l in range(L):
            v["dW" + str(l+1)] = np.zeros((parameters['W' + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1])) 
            v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],1))
        return v

    def update_parameters_with_momentum(self, parameters, grads, v):
        L = len(parameters) // 2 
        for l in range(L):
            v["dW" + str(l+1)] = self.beta*v["dW" + str(l+1)] + (1-self.beta)*grads['dW' + str(l+1)]
            v["db" + str(l+1)] = self.beta*v["db" + str(l+1)] + (1-self.beta)*grads['db' + str(l+1)]
            parameters["W" + str(l+1)] -= self.learning_rate*v["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= self.learning_rate*v["db" + str(l+1)]
        return parameters, v
    
    # GRADED FUNCTION: initialize_adam

    def initialize_adam(self, parameters) :
        L = len(parameters) // 2 
        v = {}
        s = {}
        for l in range(L):
            v["dW" + str(l+1)] = 0
            v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],1))
            s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
            s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],1))
        return v, s

    def update_parameters_with_adam(self, parameters, grads, v, s, t):
        L = len(parameters) // 2                 
        v_corrected = {}                         
        s_corrected = {}                         
        for l in range(L):
            v["dW" + str(l+1)] = self.beta1*v["dW" + str(l+1)] + (1-self.beta1)*grads["dW" + str(l+1)]
            v["db" + str(l+1)] = self.beta1*v["db" + str(l+1)] + (1-self.beta1)*grads["db" + str(l+1)]
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-self.beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-self.beta1**t)
            s["dW" + str(l+1)] = self.beta2*s["dW" + str(l+1)] + (1-self.beta2)*(grads["dW" + str(l+1)])**2
            s["db" + str(l+1)] = self.beta2*s["db" + str(l+1)] + (1-self.beta2)*(grads["db" + str(l+1)])**2
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-self.beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-self.beta2**t)
            parameters["W" + str(l+1)] -= self.learning_rate * ( v_corrected["dW" + str(l+1)] / ( np.sqrt(s_corrected["dW" + str(l+1)]) + self.epsilon ) ) 
            parameters["b" + str(l+1)] -= self.learning_rate * ( v_corrected["db" + str(l+1)] / ( np.sqrt(s_corrected["db" + str(l+1)]) + self.epsilon ) ) 
        return parameters, v, s

    def fit(self):
        costs = [] 
        # Initialize parameters dictionary.
        if self.initialization == "zeros":
            parameters = self.initialize_parameters_zeros()
        elif self.initialization == "random":
            parameters = self.initialize_parameters_random()
        elif self.initialization == "xavier":
            parameters = self.initialize_parameters_xavier()
        elif self.initialization == "he":
            parameters = self.initialize_parameters_he()
        # Initialize the optimizer
        if self.optimizer == "gd":
            pass # no self.initialization required for gradient descent
        elif self.optimizer == "momentum":
            v = self.initialize_velocity(parameters)
        elif self.optimizer == "adam":
            v, s = self.initialize_adam(parameters)
        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self.L_model_forward(self.X, parameters)
            # Compute cost.
            cost = self.compute_cost(AL)
            # Backward propagation.
            grads = self.L_model_backward(AL, caches)
            # Update parameters
            if self.optimizer == "gd":
                parameters = self.update_parameters_with_gd(parameters, grads)
            elif self.optimizer == "momentum":
                parameters, v = self.update_parameters_with_momentum(parameters, grads, v)
            elif self.optimizer == "adam":
                i = i + 1 # Adam counter
                parameters, v, s = self.update_parameters_with_adam(parameters, grads, v, s, i)
            # Print the cost every 100 training example
            if self.print_cost and i % self.nbr_print == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if self.print_cost and i % self.nbr_print == 0:
                costs.append(cost)
        if self.print_cost:
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

        return parameters
    
    def predict(self, X, parameters):
        m = X.shape[1]
        n = len(parameters) // 2
        p = np.zeros((1,m))
        probas, caches = self.L_model_forward(X, parameters)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        return p[0]