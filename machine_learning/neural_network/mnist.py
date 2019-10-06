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
        for row in csv_reader:
            if not row:
                continue
            # Almacena en la lista dataset cada fila
            dataset.append(row)
    return dataset

def generate_csv(classifications):
    # Crea el archivo results.csv y lo declara como file_csv para poder acceder a el
    with open("results.csv", "w") as file_csv:
        # En vez de imprimir en consola, imprime en file_csv
        print('ImageId,Label', file=file_csv)
        # Imprime el id del test sample y la predicción
        for i in range(len(classifications)):
            print('{},{}'.format((i+1),classifications[i]), file=file_csv)

# Convierte los valores del CSV de strings a flotantes
def str_column_to_float(dataset):
    # i es el numero de columna
    for i in range(len(dataset[0])):
        # para cada fila va a convertir unicamente la columna indicada por i (recorre en vertical)
        for row in dataset:
            row[i] = float(row[i].strip())

# Obtiene el valor maximo, el valor minimo y el promedio de cada columna para utilizarlos en la normalizacion
def dataset_minmaxavg(dataset):
    # minmaxavg es una lista donde se va a guardar el valor maximo, minimo y promedio de cada columna
    minmaxavg = list()
    for i in range(len(dataset[0])):
        # col_values almacena todos los valores de cada columna
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        value_avg = sum(col_values)/len(col_values)
        # Append agrega a la lista minmaxavg el valor minimo, el valor maximo y el valor promedio
        minmaxavg.append([value_min, value_max, value_avg])
    return minmaxavg

# Escala los valores del dataset en el rango 0 - 1
# https://kharshit.github.io/blog/2018/03/23/scaling-vs-normalization
def scale_dataset(dataset, minmaxavg, test=False):
    # Recorre de manera horizontal
    if test:
        for row in dataset:
            for i in range(len(row)):
                # nuevo_valor = (valor_original - valor_min)/(valor_max - valor_min)
                if(minmaxavg[i][0]!=minmaxavg[i][1]):
                    row[i] = (row[i] - minmaxavg[i][0]) / (minmaxavg[i][1] - minmaxavg[i][0])
                else:
                    row[i]=0.0001
    else:
        for row in dataset:
            for i in range(1, len(row)):
                # nuevo_valor = (valor_original - valor_min)/(valor_max - valor_min)
                if(minmaxavg[i][0]!=minmaxavg[i][1]):
                    row[i] = (row[i] - minmaxavg[i][0]) / (minmaxavg[i][1] - minmaxavg[i][0])
                else:
                    row[i]=0.0001
            # otra opcion podria ser normalizar
            # nuevo_valor = (valor_original - valor_promedio)/(valor_max - valor_min)
            # row[i] = (row[i] - minmaxavg[i][2]) / (minmaxavg[i][1] - minmaxavg[i][0])
            # pero habria que modificar el range de la siguiente manera range(0, len(row)-1, 1)
            # para que no modifique la ultima columna (los outputs ó Y)

def init_weights(layers):
    weights = list()
    #probar con layers[i] + layers[i+1] en el denominador
    #probar con diferente numerador, 2.0 es XAVIER method
    for i in range(len(layers)-1):
        var = math.sqrt(1.0/layers[i])
        weights.append([[random.random() * var for _ in range(layers[i])] for _ in range(layers[i+1])])
    return weights

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))

# Sigmoid derivative
def d_sigmoid(z):
    return z*(1-z)

def dot_product(inputs, weights):
    acum = 0
    for i in range(len(inputs)):
        acum += inputs[i] * weights[i]
    return acum

def feed_forward_propagation(inputs, weights, test=False):
    # Quita el lable del sample cuando es training
    if not test:
        inputs.pop(0)
    # Asigna los inputs a la primera layer
    nodes[0] = inputs
    # Recorre cada layer excepto la última
    for i in range(len(nodes) - 1):
        # Hace la multiplicacion entre la layer i y los pesos para generar los outputs de la siguiente
        # También aplica la función de activación
        for j in range(len(nodes[i+1])):
            nodes[i+1][j] = sigmoid(dot_product(nodes[i], weights[i][j]))

# Calcula el error en cada output node
def t_errors():
    total_err = 0
    # La lista errors va a almacenar el mean square error de cada output node con respecto al output esperado
    errors = list()
    # Recorre cada nodo del output layer
    for x in range(len(target)):
        # Agrega a la lista errors el error de cada nodo
        errors.append(0.5*(target[x]-nodes[-1][x])**2)
        # Se suman todos los errores para tener el error total de esa sample
        total_err += errors[x]
    print('\ntarget: ', target)
    print('prediction:\t', nodes[-1].index(max(nodes[-1])), '\n')
    print("\nOutput Errors:", errors)
    print("Total Error: ",total_err,'\n')
    return errors

def back_propagation(weights):
    # Como las deltas se van a utilizar en varias operaciones y para evitar que se calculen varias veces
    # se van a almacenar en una lista
    deltas = list()
    # Se asigna el valor del learning rate
    alpha = 0.015
    # Se calculan las deltas
    for x in range(len(errors)):
        # dE_total/d_out[x] * d_out[x]/d_net[x]
        deltas.append((-target[x]+nodes[-1][x])*(d_sigmoid(nodes[-1][x])))

    # Actualizar Hidden_Output weights
    j=0
    for delta in deltas:
        for i in range(len(nodes[1])):
            # Se almacenan en una lista de weights temporal los cuales se van a copiar a la lista de
            # weights cuando termine la propagación hacia atras
            weights_temp[1][j][i] = weights[1][j][i] - (alpha*delta*nodes[1][i])
        j+=1
    # Actualizar Input_Hidden weights
    delta_weights_Hnode=list()
    for i in range(len(weights[1][1])):
        mult=0
        # Al igual que las deltas como se van a utilizar varias veces (S1*Wx+...+Sn*Wxn)*(dout/dnet)
        # se almacenan en una lista
        for j in range(len(deltas)):
            # mult almacena la suma de las derivadas multiplicadas por cada peso(S1*Wx+...+Sn*Wxn)
            mult += deltas[j]*weights[1][j][i]
        # mult se multiplica por (dout/dnet)
        mult*=d_sigmoid(nodes[1][i])
        # (S1*Wx+...+Sn*Wxn)*(dout/dnet) se almacena en la lista delta_weights_Hnode
        delta_weights_Hnode.append(mult)
    # Se recorre cada uno de los deltas calculados
    for i in range(len(delta_weights_Hnode)):
        # Por cada delta se recorren los nodos y se actualizan los pesos temporales
        for j in range(len(nodes[0])):
            weights_temp[0][i][j] = weights[0][i][j] - (alpha*delta_weights_Hnode[i]*nodes[0][j])

    return weights_temp

# Se asigna el path donde esta el csv de entrenamiento
filename = 'train.csv'
# Se carga el archivo de entrenamiento como lista
dataset = load_csv(filename)
# Convierte los valores del archivo a flotantes para poder hacer operaciones con ellos
str_column_to_float(dataset)
# Se calculan los valores maximos y minimos de cada columna y se almacenan en una lista
minmaxavg = dataset_minmaxavg(dataset)
# Se actualiza el dataset utilizando minmax para escalar los valores de entrada a la red
scale_dataset(dataset, minmaxavg)
# Se declaran los nodos que habrá en cada capa
layers = (784, 64, 10)
# Inicializa los pesos de manera 'aleatoria' utilizando Xavier Method
weights = init_weights(layers)
# Copia los pesos en una matriz temporal para poder actualizar en back prop
weights_temp = copy.deepcopy(weights)
# Genera la matriz de los nodos separados por capas inicializandolos en 0
nodes = [[0.0 for _ in range(layers[i])] for i in range(len(layers))]
for epoch in range(100):
    # Se almacenan los valores de pixeles de cada sample
    inputs = dataset[epoch%42000]
    # Se almacena en una variable el output esperado
    label = inputs[0]
    # Se asignan los valores esperados para el sample
    target = [0.01 if x!=label else 0.99 for x in range(len(nodes[2]))]
    # Incia la propagación hacia adelante
    feed_forward_propagation(copy.deepcopy(inputs), weights)
    # Error total del epoch
    errors = t_errors()
    # Inicia la propagacion hacia atras y almacena los nuevos pesos en weights
    weights = back_propagation(weights)
    print("Epoch: ",epoch)

# Load test csv
# Se crea una lista para almacenar las predicciones
classifications = list()
# Se asigna el path donde esta el csv de prueba
test_filename = 'test.csv'
# Se carga el archivo de prueba como lista
test_dataset = load_csv(test_filename)
# Convierte los valores del archivo a flotantes para poder hacer operaciones con ellos
str_column_to_float(test_dataset)
# Se calculan los valores maximos y minimos de cada columna y se almacenan en una lista
minmaxavg = dataset_minmaxavg(test_dataset)
# Se actualiza el dataset utilizando minmax para escalar los valores de entrada a la red
scale_dataset(test_dataset, minmaxavg, True)
# Recorre cada sample
for row in test_dataset:
    # Asigna los valores iniciales de la red (Pixeles)
    inputs = row
    # Incia la propagación hacia adelante
    feed_forward_propagation(inputs, weights, True)
    # Se almacena la predicción en la lista classifications calculando el indice del
    # valor máximo de las predicciones
    classifications.append(nodes[2].index(max(nodes[2])))
# Se genera el csv con el indice del sample y su predicción
generate_csv(classifications)
