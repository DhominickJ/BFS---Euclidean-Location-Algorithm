import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
fields = ['NAMES', 'LATITUDE', 'LONGITUDE', 'CITY', 'ZIP']
df = pd.read_csv('botanicalcris.csv', skipinitialspace=True, usecols=fields)

# Define a function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    sum = 0
    for a, b in zip(point1, point2):
        sum += (a - b) ** 2
    sum = math.sqrt(sum)
    return sum

# TODO: DONE Rewrite the Whole Algorithm for Readability as Instructed
def manhattan_distance(point1, point2):
    distance = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        distance += absolute_difference
    return distance
# TODO: DONE Implemented Manual Sorting with Bubble Sort 
def bubblesort(distances):
    for i in range(len(distances)):
        for j in range(len(distances) - i - 1):
            if distances[j][1] > distances[j + 1][1]:
                distances[j], distances[j + 1] = distances[j + 1], distances[j]
    return distances

# Define a function to find the k nearest neighbors using Euclidean distance and bfs TODO: DONE Implement the Bubble Sort for Alternative Sorting Algorithm
def get_euclidean_neighbors(df, test_row, k):
    distances = []
    for index, row in df.iterrows():
        dist = euclidean_distance(test_row, [row['LATITUDE'], row['LONGITUDE']])
        distances.append((row, dist))
    distances = bubblesort(distances)
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def get_manhattan_neighbors(df, test_row, k):
    distances = []
    for index, row in df.iterrows():
        dist = manhattan_distance(test_row, [row['LATITUDE'], row['LONGITUDE']])
        distances.append((row, dist))
    distances = bubblesort(distances)
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Get the k nearest neighbors
k = 4
test_row = [50, 10]  # This should be replaced with your location to check for the nearest park
neighbors = get_manhattan_neighbors(df, test_row, k)

# Print the nearest park
print('The nearest park is: ', neighbors[0]['NAMES'])

print('The 1st nearest park is: ', neighbors[0]['NAMES'])
print('The 2nd nearest park is: ', neighbors[1]['NAMES'])
print('The 3rd nearest park is: ', neighbors[2]['NAMES'])

#Print the k nearest parks
for i in range(k):
    if k > len(neighbors):
        print('There are only ', len(neighbors), ' parks in the dataset.')
        break

# Create a graph object
G = nx.Graph()

# Add nodes (parks) to the graph
for neighbor in neighbors:
    G.add_node(neighbor['NAMES'])

# Add edges (distances) between parks
for i in range(len(neighbors)):
    for j in range(i + 1, len(neighbors)):
        distance = euclidean_distance([neighbors[i]['LATITUDE'], neighbors[i]['LONGITUDE']],
                                      [neighbors[j]['LATITUDE'], neighbors[j]['LONGITUDE']])
        G.add_edge(neighbors[i]['NAMES'], neighbors[j]['NAMES'], weight=distance)

# Draw the graph
pos = nx.kamada_kawai_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels) # Use this in order to create show the distances of gardens
nx.draw_networkx_nodes(G, pos, node_color='cyan', node_size=500)
nx.draw_networkx_nodes(G, pos, nodelist=[neighbors[0]['NAMES']], node_color='red', node_size=500)
plt.show()

