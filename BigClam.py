import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
# from agm import AGM, community
import numpy as np
import random

def bigclam(graph, community_num, step_size=0.01,threshold=0.01): # max_iter
	# initialize a random matrix F
	F = np.matrix(np.ones((len(graph.nodes()),community_num)))
	print np.shape(F)
	for i in range(np.shape(F)[0]):
		for j in range(np.shape(F)[1]):
			F[i,j] = random.random()

	# MLE
	iter = 0
	while  True:
		# we cannot use last_F = F cause this is an assignment and will pass the reference of the class F, not copy
		last_F = F.copy()
		iter += 1
		for node in range(len(graph.nodes())):
			gradient = np.matrix(np.zeros((1,community_num)))
			u = F[node,:]
			for neighbor in graph.neighbors(node):
				v = F[neighbor,:]
				gradient += v*float(np.exp(-u*v.transpose())/(1-np.exp(-u*v.transpose()))) + v
			gradient += u - np.sum(F,axis=0)
			# update row u
			F[node,:] = F[node,:] + step_size*gradient
			# check for non-negative constrain
			for i in range(community_num):
				if F[node,i] < 0:
					F[node,i] = 0
		print 'iter is ', iter
		change = np.sum(np.multiply(F-last_F,F-last_F))
		if  change <= 0.01:
			break
		print 'change is ', float(change)
	print iter
	return F

def GenerateGraph():
	G = nx.erdos_renyi_graph(200, 0.01)
	# 4 communities 
	community_1 = range(20)+range(50,80)+range(110,140)
	community_2 = range(18,68)+range(150,180)
	community_3 = range(23)+range(76,123)
	community_4 = range(160,200)+range(30,66)
	communities = [community_1,community_2,community_3,community_4]
	for community in communities:
		random.shuffle(community)
		for i in range(230):
			pair = random.sample(community, 2)
			G.add_edge(pair[0], pair[1])
	return G

G = GenerateGraph()
F = bigclam(G, 4)
print F


# nodes = range(24)
# communities = [community(range(12), 0.8),
#                community(range(6,17), 0.8),
#                community(range(8,12)+range(17,24), 0.8)]
# G = AGM(nodes,communities)
# F = bigclam(G, 3)
# print F


			


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

