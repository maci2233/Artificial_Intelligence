import random
import math
import matplotlib.pyplot as plt

def init_weights(layers):
    weights = list()
    #probar con layers[i] + layers[i+1] en el denomidor
    #probar con diferente numerador, 2.0 es XAVIER method
    for i in range(len(layers)-1):
        var = math.sqrt(2.0/layers[i])
        weights.append([[random.random() * var for _ in range(layers[i])] for _ in range(layers[i+1])])
    return weights

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))

def dot_product(inputs, weights):
    acum = 0
    for i in range(len(inputs)):
        acum += inputs[i] * weights[i]
    return acum

def feed_forward_propagation(inputs, weights):
    nodes[0] = inputs
    for i in range(len(nodes) - 1):
        for j in range(len(nodes[i+1])):
            nodes[i+1][j] = sigmoid(dot_product(nodes[i], weights[i][j]))

if __name__ == '__main__':
    layers = (784, 64, 10)
    weights = init_weights(layers)
    input = [8.5, 0.65, 1.2]
    nodes = [[0.0 for _ in range(layers[i])] for i in range(len(layers))]
    prediction = feed_forward_propagation(input, weights)
