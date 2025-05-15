# Installing Libraries
import numpy as np
from random import seed
import pprint

# Initializing the pprint module
pp = pprint.PrettyPrinter(depth=4)

# Generating a basic neural network with 3 layers; an input layer with 2 inputs, a hidden layer with 2 nodes, and an output layer with 1 node

def basic_neural_network():
    # Function that returns the weighted sum given inputs, weights and bias
    def weightedSum(x1, x2, w1, w2, b):
        return (x1*w1) + (x2*w2) + b

    # Function that returns activation value given weighted sum
    def activation(z):
        return 1.0 / (1.0 + np.exp(-z))

    # Generating weights and biases
    # weights = np.around(np.random.uniform(size=6), decimals=2)
    # biases = np.around(np.random.uniform(size=3), decimals=2)
    weights = np.array([0.69, 0.24, 0.5, 0.84, 0.61, 0.05])
    biases = np.array([0.37, 0.59, 0.68])

    print(weights)
    print(biases)

    # Setting inputs x_1 and x_2
    x_1 = 0.5
    x_2 = 0.85

    # Calculating the weighted sums of the nodes in the hidden layer
    z_11 = weightedSum(x_1, x_2, weights[0], weights[1], biases[0])
    z_12 = weightedSum(x_1, x_2, weights[2], weights[3], biases[1])

    # Printing the weighted sums in the hidden layer
    print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(np.around(z_11, decimals=4)))
    print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))
    print()

    # Calcuating the activation of the nodes in the hidden layer
    a_11 = activation(z_11)
    a_12 = activation(z_12)


    # Printing the activation of the nodes in the hidden layer
    print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
    print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))
    print()

    # Calculating the weighted sum of the node in the output layer and printing it
    z_2 = weightedSum(a_11, a_12, weights[4], weights[5], biases[2])
    print('The weighted sum of the input at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

    # Calculating the activation of the node in the output layer and printing it
    a_2 = activation(z_2)
    print('The activation of the node in the output layer is {}'.format(np.around(a_2, decimals=4)))

#####

# Generating a more complex neural network

def generalized_neural_network():
    # Formal definition of the structure
    n = 2 # number of inputs
    num_hidden_layers = 2 # number of hidden layers
    m = [2,2] # number of nodes in each hidden layer
    num_nodes_output = 1 # number of output nodes

    # Initializing weights and biases

    num_nodes_previous = n # Number of nodes in the previous layer

    network = {} # Initializing an empty dictionary

    # Loop through each layer and randomly initialize the weights and biases associated with each node; take note how we add 1 to the number of hidden layers to include the output layer
    for layer in range(num_hidden_layers + 1):
        # Determine name of layer
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes= num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = m[layer]

        # Initialize weights and biases associated to each node in the layer
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    print(network)

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):

    num_nodes_previous = num_inputs # Number of nodes in the previous layer

    network = {} # Initializing an empty dictionary

    # Loop through each layer and randomly initialize the weights and biases associated with each node; take note how we add 1 to the number of hidden layers to include the output layer
    for layer in range(num_hidden_layers + 1):
        # Determine name of layer
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes= num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = num_nodes_hidden[layer]

        # Initialize weights and biases associated to each node in the layer
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network

# Generating a small neural network
small_network = initialize_network(5, 3, [3,2,3], 1)

# Function to compute the weighted sum of a node based on inputs, weights, and bias
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

# Function to calcuate activation of a node based on weighted sum
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

# Generating 5 inputs to input into small_network
np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

# Printing the layers, weights, and biases
print('The values of the neural network are:')
pprint(small_network)
print()

# Printing the network inputs
print('The inputs to the network are {}'.format(inputs))
print()


# Setting up forward propagation
def forward_propagation(network, inputs):
    layer_inputs = list(inputs) #  Starting with the input layer

    for layer in network:
        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:

            node_data = layer_data[layer_node]

            # Computing the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(float(np.around(node_output[0], decimals=4)))
        
        if layer != 'output':
            print('The outputs of the nodes in the hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
        
        layer_inputs = layer_outputs # setting the output of this layer as the inputs of the next

    print() # Adding a space after all outputs of the layers are printed
    network_predictions = layer_outputs
    return network_predictions

predictions = forward_propagation(small_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))
