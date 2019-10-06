import random
import math
import copy
from csv import reader
#import matplotlib.pyplot as plt

def load_csv(filename):
    dataset = list()
    # Abre el archivo en modo lectura
    with open(filename, 'r') as file:
        # Lee el archivo y lo almacena en una variable
        csv_reader = reader(file)
        # Recorre cada linea del archivo
        first = True
        for row in csv_reader:
            if first:
                first = False
                continue
            if not row:
                continue
            # Almacena en la lista dataset cada fila
            dataset.append(row)
    return dataset

def generate_csv(classifications):
    with open("results.csv", "w") as file_csv:
        print('ImageId,Label', file=file_csv)
        for i in range(len(classifications)):
            print((i+1), ',', classifications[i], file=file_csv)

# Convierte los valores del CSV de strings a flotantes
def str_column_to_float(dataset):
    # i es el numero de columna
    for i in range(len(dataset[0])):
        # para cada fila va a convertir unicamente la columna indicada por i (recorre en vertical)
        for row in dataset:
            row[i] = float(row[i].strip())

def init_weights(layers):
    weights = list()
    #probar con layers[i] + layers[i+1] en el denominador
    #probar con diferente numerador, 2.0 es XAVIER method
    for i in range(len(layers)-1):
        var = math.sqrt(2.0/layers[i])
        #weights.append([[float(input()) for _ in range(layers[i])] for _ in range(layers[i+1])])
        weights.append([[random.random() * var for _ in range(layers[i])] for _ in range(layers[i+1])])
    return weights

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))

def d_sigmoid(z):
	return z*(1-z)

def dot_product(inputs, weights):
    acum = 0
    for i in range(len(inputs)):
        acum += inputs[i] * weights[i]
    return acum

def feed_forward_propagation(inputs, weights):
    nodes[0] = inputs
    # bias1 = 0.35
    #print(nodes[2])
    for i in range(len(nodes) - 1):
        for j in range(len(nodes[i+1])):
        	nodes[i+1][j] = sigmoid(dot_product(nodes[i], weights[i][j]))
    return nodes[2]
'''
            if(i==0):
                nodes[i+1][j] = sigmoid(dot_product(nodes[i], weights[i][j])+bias1)
            else:
            	nodes[i+1][j] = sigmoid(dot_product(nodes[i], weights[i][j])+bias2)
'''

def feed_forward_propagation_test(inputs, weights):
    new_nodes = [[0.0 for _ in range(layers[i])] for i in range(len(layers))]
    new_nodes[0] = inputs
    # bias1 = 0.35
    #print(nodes[2])
    for i in range(len(nodes) - 1):
        for j in range(len(nodes[i+1])):
        	new_nodes[i+1][j] = sigmoid(dot_product(new_nodes[i], weights[i][j]))
    return new_nodes[2]


# Calcula el error en cada output node
def t_errors():
    total_err = 0
    errors = list()
    for x in range(len(target)):
        errors.append(0.5*(target[x]-nodes[len(nodes)-1][x])**2)
        total_err += errors[x]
    #print("Output Errors:", errors)
    #print("Total Error: ",total_err)
    return errors

def back_propagation(weights):
    deltas = list()
    alpha = 0.05
    for x in range(len(errors)):
        # dE_total/dout_node * dout_node/dnet_node
         deltas.append((-target[x]+nodes[len(nodes)-1][x])*(d_sigmoid(nodes[len(nodes)-1][x])))

    # Update Hidden_Output weights
    j=0
    for delta in deltas:
    	for i in range(len(nodes[1])):
    		weights_temp[1][j][i] = weights[1][j][i] - (alpha*delta*nodes[1][i])
    	j+=1
    # Update Input_Hidden weights
    delta_weights_Hnode=list()
    for i in range(len(weights[1][1])):
    	mult=0
    	# Build a list with (S1*Wx+...+Sn*Wxn)*(dout/dnet)
    	for j in range(len(deltas)):
    		mult += deltas[j]*weights[1][j][i]
    	mult*=d_sigmoid(nodes[1][i])
    	delta_weights_Hnode.append(mult)

    for i in range(len(delta_weights_Hnode)):
    	for j in range(len(nodes[0])):
    		weights_temp[0][i][j] = weights[0][i][j] - (alpha*delta_weights_Hnode[i]*nodes[0][j])

    # weights = copy.deepcopy(weights_temp)
    return weights_temp
    # print(weights)
    '''for i in range(len(weights_temp[0])):
    	for j in range(len(weights_temp[0][0])):
    		weights_temp[0][i][j] =
    weights_temp[0][0][0] = ((deltas[0]*weights[1][0][0])+(deltas[1]*weights[1][1][0]))*d_sigmoid(nodes[1][0])*nodes[0][0]
    weights_temp[0][0][1] = ((deltas[0]*weights[1][0][0])+(deltas[1]*weights[1][1][0]))*d_sigmoid(nodes[1][0])*nodes[0][1]
    weights_temp[0][1][0] = ((deltas[0]*weights[1][0][1])+(deltas[1]*weights[1][1][1]))*d_sigmoid(nodes[1][1])*nodes[0][0]
    weights_temp[0][1][1] = ((deltas[0]*weights[1][0][1])+(deltas[1]*weights[1][1][1]))*d_sigmoid(nodes[1][1])*nodes[0][1]
    weights = copy.deepcopy(weights_temp)
    print(weights)'''


filename = 'train.csv'
dataset = load_csv(filename)
str_column_to_float(dataset)
layers = (784, 64, 10)
# Inicializa los pesos de manera aleatoria
weights = init_weights(layers)
# Copia los pesos en una matriz temporal para poder actualizar en back prop
weights_temp = copy.deepcopy(weights)
# Genera la matriz de los nodos separados por capas
nodes = [[0.0 for _ in range(layers[i])] for i in range(len(layers))]
for epoch in range(100):
    print("epoch " + str(epoch))
    # Valores de pixeles de cada sample
    inputs = dataset[epoch % 10000]
    # Output esperado(target)
    label = inputs[0]
    inputs.pop(0)
    # Valores esperados para el sample
    target = [0.01 if x!=label else 0.99 for x in range(len(nodes[2]))]
    # Genera la estructura de la lista de matrices
    predictions = feed_forward_propagation(inputs, weights)
    print(predictions)
    errors = t_errors()
    weights = back_propagation(weights)

# Load test csv

test_filename = 'test.csv'
test_dataset = load_csv(test_filename)
str_column_to_float(test_dataset)
classifications = list()
for row in test_dataset:
    inputs = row
    predictions = feed_forward_propagation_test(inputs, weights)
    print("predictions:")
    print(predictions)
    classifications.append(max(predictions))
	#print(test_nodes[2])
	#print(max(test_nodes[2]))
generate_csv(classifications)
