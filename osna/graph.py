# -*- coding: utf-8 -*-

"""Plot graphs"""
import networkx as nx
import matplotlib.pyplot as plt

def read_edge_list(file_path):
    """
    Read edges from a file to generate graph
    """
#     sample:
#     urllib.request.urlretrieve("http://snap.stanford.edu/data/ca-GrQc.txt.gz", "ca-GrQc.txt.gz")
#     graph = nx.read_edgelist('ca-GrQc.txt.gz')
    graph = nx.read_edgelist(file_path)
    return graph

def get_subgraph(graph, nodes, n=100):
    """
    Get the subgraph consisting of a list node and their neighbors,
    plus their neighbors' neighbors, up to $n$ total nodes
    """
    neighbors = set()
    for ni in nodes:
        neighbors |= set(graph.neighbors(ni))
    # plot at least the target node and his neighbors.
    result = set(nodes) | neighbors
    # add "friends of friends" up to n total nodes.
    for x in neighbors:
        # how many more nodes can we add?
        maxsize = n - len(result) 
        toadd = set(graph.neighbors(x)) - result
        result.update(list(toadd)[:maxsize])
        if len(result) > n:
            break
    return graph.subgraph(result)

def plot_graph(subgraph, target_nodes):
    """
    Plot this subgraph of nodes, coloring the specified list of target_nodes in red.
    """
    nodes = list(subgraph.nodes())
    colors = ['b'] * len(nodes)
    for n in target_nodes:
        idx = nodes.index(n)
        colors[idx] = 'r'
    sizes = [800] * len(nodes)
    sizes[idx] = 1000
    plt.figure(figsize=(10,10))
    plt.axis('off')
    nx.draw_networkx(subgraph, nodelist=nodes, with_labels=True,
                     width=.5, node_color=colors,
                     node_size=sizes, alpha=.5)
