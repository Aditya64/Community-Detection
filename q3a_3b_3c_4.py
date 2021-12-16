#girvan newman clustering
import itertools
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_comm
import time
import warnings
warnings.filterwarnings("ignore")

iteration = 1

#Read graph information from the dataset given and load graphs
Graph1 = nx.read_gml("karate.gml", label = 'id')
Graph2 = nx.read_weighted_edgelist("jazz.net")
Graph3 = nx.read_gml("dolphins.gml", label = 'id')

#initialize time values
times = [[0,0,0],[0,0,0],[0,0,0]]
for t in range(iteration):          #run the algorithm for the given number of iterations to get the average time data
    print("************KARATE CLUB NETWORK************")
    print("************GIRVAN_NEWMAN**************")
    k = 3   # k is the number of iterations of Girvan newman algorithm.
    initial_time = time.time()
    comp = girvan_newman(Graph1) 
    # comp is an iterator over tuples of sets of nodes in Graph being selected. Each set of node is a community, each tuple is a       sequence of communities at a particular level of the algorithm.
    
    times[0][0] +=  time.time()-initial_time # here we are calculating the run time of the algorithm
    print("Girvan Newman Time",times[0][0]/iteration)
    
    cluster = [] #stores the communities
    for communities in itertools.islice(comp, k):
        cluster = list(sorted(c) for c in communities)
    print("Number of clusters : ", len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph1, cluster))

    print("*****************MODULARITY_MAXIMIZATION************")
    initial_time = time.time()
    cluster = list(greedy_modularity_communities(Graph1))#stores the communities observed after modularity maximization algorithm
    times[0][1] += time.time()-initial_time
    print("Modularity Maximization Time",times[0][1]/iteration)
    print("Number of clusters : ", len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph1, cluster))

    print("**************SPECTRAL*************")
    initial_time = time.time()
    adjacent_mat = nx.to_numpy_matrix(Graph1) 
    # we are storing the graph adjacency matrix as a numpy matrix and it is being passed as a parameter to the spectral                 clustering algorithm in the next step.
    sc = SpectralClustering(3, affinity='precomputed', n_init=100)
    sc.fit(adjacent_mat)
    times[0][2] += time.time()-initial_time
    print("Spectral Clustering Time",times[0][2]/iteration)

    num = len(set(sc.labels_))
    print("Number of clusters : ", num)
    cluster = []
    for i in range(num):
        cluster.append([])
    for i , val in enumerate(sc.labels_):
        cluster[val].append(i+1)

    print("Modularity score : ",nx_comm.modularity(Graph1, cluster))
    print("\n")
    
    print("*****************JAZZ MUSICIANS NETWORK******************")
    print("***************GIRVAN_NEWMAN*******************")
    k=5
    initial_time = time.time()
    comp = girvan_newman(Graph2)
    times[1][0] += time.time()-initial_time
    print("Girvan Newman Time",times[1][0]/iteration)
    for communities in itertools.islice(comp, k):
        cluster = list(sorted(c) for c in communities)

    print("Number of clusters : ", len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph2, cluster))

    print("***********MODULARITY_MAXIMIZATION*******************")
    initial_time = time.time()
    cluster = list(greedy_modularity_communities(Graph2))
    times[1][1] += time.time()-initial_time
    print("Modularity Maximization Time",times[1][1]/iteration)
    print("Number of clusters : ", len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph2, cluster))

    print("****************SPECTRAL*******************")
    initial_time = time.time()
    adjacent_mat = nx.to_numpy_matrix(Graph2)
    sc = SpectralClustering(4, affinity='precomputed', n_init=100)
    sc.fit(adjacent_mat)
    times[1][2] += time.time()-initial_time
    print("Spectral Clustering Time",times[1][2]/iteration)
    lg = list(Graph2)
    num = len(set(sc.labels_))
    print("Number of clusters : ",num)
    cluster = []
    for i in range(num):
        cluster.append([])
    for i , val in enumerate(sc.labels_):
        cluster[val].append(lg[i])

    print("Modularity score : ",nx_comm.modularity(Graph2, cluster))
    print("\n")

    print("*****************DOLPHINS SOCIAL NETWORK**************")
    print("****************GIRVAN_NEWMAN****************")
    k=4
    initial_time = time.time()
    comp = girvan_newman(Graph3)
    times[2][0] += time.time()-initial_time
    print("Girvan Newman Time",times[2][0]/iteration)
    for communities in itertools.islice(comp, k):
        cluster = list(sorted(c) for c in communities)

    print("Number of clusters : ", len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph3, cluster))

    print("******************MODULARITY_MAXIMIZATION*****************")
    initial_time = time.time()
    cluster = list(greedy_modularity_communities(Graph3))
    times[2][1] += time.time()-initial_time
    print("Modularity Maximization Time",times[2][1]/iteration)
    print("Number of clusters : ",len(cluster))
    print("Modularity score : ",nx_comm.modularity(Graph3, cluster))

    print("**************SPECTRAL*****************")
    initial_time = time.time()
    adjacent_mat = nx.to_numpy_matrix(Graph3)
    sc = SpectralClustering(4, affinity='precomputed', n_init=100)
    sc.fit(adjacent_mat)
    times[2][2] += time.time()-initial_time
    print("Spectral Clustering Time ",times[2][2]/iteration)
    lg = list(Graph3)
    num = len(set(sc.labels_))
    print("Number of clusters : ", num)
    cluster = []
    for i in range(num):
        cluster.append([])
    for i , val in enumerate(sc.labels_):
        cluster[val].append(lg[i])
    print("Modularity score : ",nx_comm.modularity(Graph3, cluster))
    print("\n")
