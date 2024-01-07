import csv
import numpy as np
import re
from scipy.sparse import csr_matrix, csgraph
from scipy.linalg import eig

#Creating 50*50 empty 2D array
graph = np.zeros([50, 50], dtype = int)
Ri=[]
#Opening the csv file
with open('sample_clustered.csv') as file:
    reader = csv.reader(file)
    # displaying the contents of the CSV file
    header = next(reader)
    print('Header:',header)
    for line in reader:
        if len(line) != 0:
            print(line)
            Ri.append(float(line[2]))
            src = int(re.findall(r'[0-9]+', line[0])[0])
            dest = int(re.findall(r'[0-9]+', line[3])[0])
            location1 = [src,dest]
            location2 = [dest,src]
            #print(src, dest, location1, location2)
            graph[tuple(location1)] = 1
            graph[tuple(location2)] = 1

#print(graph)
np.save('graph.npy', graph)
print("saved")
degree_matrix = np.diag(np.sum(graph, axis=1))
laplacian_matrix = degree_matrix - graph
print(laplacian_matrix)
eigenvalues, eigenvectors = eig(laplacian_matrix)
feature_matrix = np.real(eigenvectors[:, :3])
print("Feature Matrix:\n",feature_matrix)


# Define the discount factor gamma
gamma = 0.9

# Compute f'
f_prime = np.roll(feature_matrix, -1, axis=0)
f_prime[-1] = 0

# Compute the matrix inside the inverse
A = np.dot(feature_matrix.T, feature_matrix - gamma * f_prime)

# Compute the weights
w = np.dot(np.linalg.inv(A), np.dot(feature_matrix.T, Ri))



print("Weights:", w)
