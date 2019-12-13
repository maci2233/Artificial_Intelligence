import numpy as np
import matplotlib.pyplot as plt
import math

def rand(low, high):
    return np.random.uniform(low, high)


def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i])**2 for i in range(len(p1))))


def init_centroids(K, data):
    '''
    We create "K" centroids, for each of them we will choose a random point
    from our dataset and the initial coordinate of the centroid will be equal
    to the randomly chosen point coordinate
    '''
    centroids = []
    n = len(data)
    for _ in range(K):
        rand_point = np.random.randint(0, n)
        centroids.append([data[rand_point][0], data[rand_point][1]])
    return centroids


def assign_pts_to_centroids(data, centroids):
    '''
    We iterate through all the points and for each of them we calculate the
    euclidean distance between it and each centroid. We are going to store in
    an array "c" the index of the closest centroid for each point so
    basically len(c) = len(data)
    '''
    c = []
    for point in data:
        euc_dists = [euclidean_distance(point, cent) for cent in centroids]
        c.append(euc_dists.index(min(euc_dists)))
    return c


def move_centroids(data, centroids, c):
    k = len(centroids)
    acums = [[0,0,0] for _ in range(k)] #acums[0] -> X Sum, acums[1] -> Y sum, acums[2] -> points assigned to the centroid
    for point, i in zip(data, c): #For each point and the index of its closest centroid
        acums[i][0] += point[0]
        acums[i][1] += point[1]
        acums[i][2] += 1
    for i in range(k):
        centroids[i][0] = acums[i][0] / acums[i][2]
        centroids[i][1] = acums[i][1] / acums[i][2]
    return centroids


pts_group = 25 #Points created for each group
colors = ["red", "blue", "green", "black", "pink"] * 2

g1 = np.array([[rand(0,5), rand(0,5)] for _ in range(pts_group)])
g2 = np.array([[rand(5,10), rand(5,10)] for _ in range(pts_group)])
g3 = np.array([[rand(10,15), rand(10,15)] for _ in range(pts_group)])

data = np.concatenate((g1, g2, g3)) #Data contains all the points now

plt.scatter(data[:, 0], data[:, 1])
plt.show()

K = 3  #Number of clusters
epochs = 8
plot_epochs = False
centroids = np.array(init_centroids(K, data))

'''
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=220)
plt.show()
'''

for epoch in range(epochs):
    c = assign_pts_to_centroids(data, centroids) #c array contains the index of the closest centroid for each point
    if plot_epochs:
        for point, i in zip(data, c):
            plt.scatter(point[0], point[1], c=colors[i])
        plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=220)
        plt.show()
    centroids = move_centroids(data, centroids, c)

if not plot_epochs:
    for point, i in zip(data, c):
        plt.scatter(point[0], point[1], c=colors[i])
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=220)
    plt.show()








print("Done")
