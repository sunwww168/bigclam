import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random

def bigclam(graph, community_num, step_size,threshold): # max_iter
	# initialize a random matrix F
	F = np.matrix(np.ones((len(graph.nodes(),community_num)))
	for i in range(np.shape(F)[0]):
		for j in range(np.shape(F)[1]):
			F[i][j] = random.random()

	# MLE
	iter = 0
	while  True:
		last_F = F
		iter += 1
		for node in range(len(graph.nodes())):
			gradient = np.matrix(np.zeros((1,community_num)))
			u = F[node,:]
			for neighbor in graph.neighbors(node):
				v = F[neighbor,:]
				gradient += v*float(np.exp(-u*v.transpose())/(1-exp(-u*v.transpose()))) + v
			gradient += u - np.sum(F,axis=0)
			# update row u
			F[node,:] = F[node,:] + step_size*gradient
			# check for non-negative
			for i in range(community_num):
				if F[node,i] < 0:
					F[node,i] = 0

			







# BigClam's Network Generation Model:
	# generate graph from Community membership strength Matrix F
def GenerateGraphFromCMS(F):
	# create an empty graph
	G = nx.Graph()
	# add nodes into the graph, shape(F)[0] is the # of columns of F
	G.add_nodes_from(range(np.shape(F)[0]))
	# generate edges
	for pairs in combinations(G.nodes(), 2):
		[u,v] = pairs
		# calculate link prob
		prob = 1 - np.exp(-F[u,:]*F[v,:].transpose())
		if random.random() <= prob:
			G.add_edge(u, v)
	return G

