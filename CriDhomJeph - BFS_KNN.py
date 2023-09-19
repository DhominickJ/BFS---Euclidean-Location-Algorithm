import pandas as pd
import math
import networkx as nx
import matplotlib.pyplot as plt

# Load the data
fields = ['NAME', 'LATITUDE', 'LONGITUDE']
df = pd.read_csv('gardens.csv', skipinitialspace=True, usecols=fields)

# Define a function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

# Define a function to find the k nearest neighbors using Euclidean distance and bfs
def get_neighbors(df, test_row, k):
    distances = []
    for index, row in df.iterrows():
        dist = euclidean_distance(test_row, [row['LATITUDE'], row['LONGITUDE']])
        distances.append((row, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Get the k nearest neighbors
k = 3
test_row = [50, 10]  # This should be replaced with your location to check for the nearest park
neighbors = get_neighbors(df, test_row, k)

# Print the nearest park
print('The nearest park is: ', neighbors[0]['NAME'])

print('The 1st nearest park is: ', neighbors[0]['NAME'])
print('The 2nd nearest park is: ', neighbors[1]['NAME'])
print('The 3rd nearest park is: ', neighbors[2]['NAME'])

#Print the k nearest parks
for i in range(k):
    if k > len(neighbors):
        print('There are only ', len(neighbors), ' parks in the dataset.')
        break
    # print('The ', i+1, ' nearest park is: ', neighbors[i]['NAME'])

# G = nx.Graph()

# for index, row in df.iterrows():
#     G.add_node(row['NAME'], pos=(row['LATITUDE'], row['LONGITUDE']))
# for index, row in df.iterrows():
#     for i in range(len(neighbors)):
#         if row['NAME'] == neighbors[i]['NAME']:
#             G.add_edge(test_row[0], row['NAME'], weight=i+1)

#     pos = nx.get_node_attributes(G, 'pos')
#     nx.draw(G, pos, with_labels=True)
#     labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# plt.show()

# Create a graph object
G = nx.Graph()

# Add nodes (parks) to the graph
for neighbor in neighbors:
    G.add_node(neighbor['NAME'])

# Add edges (distances) between parks
for i in range(len(neighbors)):
    for j in range(i + 1, len(neighbors)):
        distance = euclidean_distance([neighbors[i]['LATITUDE'], neighbors[i]['LONGITUDE']],
                                      [neighbors[j]['LATITUDE'], neighbors[j]['LONGITUDE']])
        G.add_edge(neighbors[i]['NAME'], neighbors[j]['NAME'], weight=distance)

# Draw the graph
# pos = nx.(G)
# pos = nx.nodes(G)
# pos = nx.fruchterman_reingold_layout (G)
pos = nx.kamada_kawai_layout(G)
# pos = nx.shell_layout(G)
# pos = nx.spring_layout(G)
# node_colors = [node['color'] if 'color' in node else 'blue' for node in G.nodes.values()]  # Set node colors
# labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
nx.draw_networkx_nodes(G, pos, node_color='cyan', node_size=500)
nx.draw_networkx_nodes(G, pos, nodelist=[neighbors[0]['NAME']], node_color='red', node_size=500)
plt.show()

