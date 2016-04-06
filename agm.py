import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

class community():
	"""community class definition"""
	def __init__(self, nodes, prob):
		self.nodes = nodes
		self.prob = prob

# AGM generates graph from communities, Matrix F is 0/1
def AGM(nodes,communities):
	# new an empty graph
	G = nx.Graph()
	# add nodes into the graph
	G.add_nodes_from(nodes)
	# generate edges
	for c in communities:
		# combinations combines #param2 nodes from #para1 for all the cases 
		for pairs in combinations(c.nodes, 2):
			if random.random() <= c.prob:
				G.add_edge(pairs[0], pairs[1])
	return G

# generate 3 communities
nodes = range(24)
community_1 = community(nodes[:12], 0.8)
community_2 = community(nodes[6:17], 0.8)
community_3 = community(nodes[8:12]+nodes[17:], 0.8)
G = AGM(nodes, [community_1,community_2,community_3])
nx.draw(G)
plt.show() 