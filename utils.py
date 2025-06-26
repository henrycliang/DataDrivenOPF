import torch
import numpy as np
import networkx as nx
from itertools import combinations

from pypower.ext2int import ext2int1
from pypower.makeYbus import makeYbus

def Cal_SystemY(mpc):
    bus=mpc['bus']
    branch=mpc['branch']
    gen=mpc['gen']
    baseMVA=mpc['baseMVA']
    _,bus,gen,branch=ext2int1(bus,gen,branch)
    Ybus,Yf,Yt=makeYbus(baseMVA,bus,branch)

    return Ybus

class LinearGradient:
    def __init__(self,mpc):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.mpc = mpc
        self.gen = mpc['gen']
        self.bus = mpc['bus']
        self.bus_num = self.bus.shape[0]
        self.gen_num = self.gen.shape[0] - 1

        self.build_graph(mpc)

    def build_graph(self,mpc):
        ## This function build the graph to calculate the linear gradient
        branch=mpc['branch']
        bus = mpc['bus']
        gen = mpc['gen']
        G = nx.Graph()

        for i in range(branch.shape[0]):
            node1 = 'B' + str(int(branch[i, 0]))
            node2 = 'B' + str(int(branch[i, 1]))
            G.add_edge(node1, node2, Num_phase=1, z=np.zeros((1), dtype=complex))
        print("The graph is tree?", nx.is_tree(G))

        ## build the index
        AllNodeNames = bus[:, 0]
        self.IndexToNodeName = {}
        self.NodeNameToIndex={}
        i = 0
        for name in AllNodeNames:
            self.NodeNameToIndex['B'+str(int(name))]=i
            self.IndexToNodeName[i] = 'B' + str(int(name))
            i += 1
        self.IndexToGenName={}
        for i in range(1,gen.shape[0]):
            self.IndexToGenName[i] = 'B' + str(int(gen[i, 0]))


        ## Appoint the root bus, and enforce it as directed tree
        G_directed = nx.DiGraph()
        for i in range(branch.shape[0]):
            node1 = 'B' + str(int(branch[i, 0]))
            node2 = 'B' + str(int(branch[i, 1]))
            path1 = nx.shortest_path_length(G, source='B1', target=node1)
            path2 = nx.shortest_path_length(G, source='B1', target=node2)
            if path1 < path2:
                G_directed.add_edge(node1, node2)
            else:
                G_directed.add_edge(node2, node1)

        ## assign the impedance to the links of the graph
        Ymatrix = Cal_SystemY(mpc)  # calculate the Y matrix
        for edge in G.edges:
            row = self.NodeNameToIndex[edge[0]]
            col = self.NodeNameToIndex[edge[1]]
            temp = [[Ymatrix[row, col]]]
            temp = -np.linalg.pinv(temp)
            G[edge[0]][edge[1]]['z'] = temp[0][0]

        ## Compute the to source impedance
        ToSource = {}
        node_set = list(G.nodes)
        node_set.remove('B1')
        for node in node_set:
            paths = nx.shortest_path(G, source='B1', target=node)
            temp = np.zeros((1), dtype=complex)
            for i in range(len(paths) - 1):
                temp += G[paths[i]][paths[i + 1]]['z']
            ToSource[node] = temp
        ToSource['B1'] = np.zeros((1), dtype=complex)

        ##
        G_dic=self.MakeDict(G_directed)

        ## calculate the gradientMat
        self.gradientMat_p,self.gradientMat_q=self.CalGradientMat(G_dic,ToSource)
        self.gradientMat_p=torch.FloatTensor(self.gradientMat_p).to(self.device)
        self.gradientMat_q=torch.FloatTensor(self.gradientMat_q).to(self.device)

    def MakeDict(self,G_n):
        ### This function to find the common node on the path to sourcebus
        G_directed = nx.DiGraph()
        G_directed.add_edges_from(G_n.edges)
        G_dic = {}
        i = 0
        for node in G_n.nodes:
            G_dic[node + node] = node
            children = list(G_directed.successors(node))
            if len(children) == 1:
                tree_1 = list(dict(nx.bfs_successors(G_directed, children[0])).values())
                tree_1_flat = [item for sublist in tree_1 for item in sublist]
                tree_1_flat.append(children[0])
                for succ in tree_1_flat:
                    i += 1
                    if node >= succ:  # bigsmall
                        G_dic[node + succ] = node
                    else:
                        G_dic[succ + node] = node
            elif len(children) == 0:
                continue
            else:
                for child in children:
                    tree_1 = list(dict(nx.bfs_successors(G_directed, child)).values())
                    tree_1_flat = [item for sublist in tree_1 for item in sublist]
                    tree_1_flat.append(child)
                    for succ in tree_1_flat:
                        i += 1
                        if node >= succ:  # bigsmall
                            G_dic[node + succ] = node
                        else:
                            G_dic[succ + node] = node
                comb = combinations(children, 2)
                for nodes in list(comb):
                    tree_1 = list(dict(nx.bfs_successors(G_directed, nodes[0])).values())
                    tree_1_flat = [item for sublist in tree_1 for item in sublist]
                    tree_1_flat.append(nodes[0])
                    tree_2 = list(dict(nx.bfs_successors(G_directed, nodes[1])).values())
                    tree_2_flat = [item for sublist in tree_2 for item in sublist]
                    tree_2_flat.append(nodes[1])
                    for node_1 in tree_1_flat:
                        for node_2 in tree_2_flat:
                            i += 1
                            if node_1 >= node_2:  # bigsmall
                                G_dic[node_1 + node_2] = node
                            else:
                                G_dic[node_2 + node_1] = node
        return G_dic

    def CalGradientMat(self,G_dic,ToSource):
        gradientMat_p = np.zeros((self.gen_num, self.bus_num), dtype=float)
        gradientMat_q = np.zeros((self.gen_num, self.bus_num), dtype=float)
        for i in range(self.gen_num):
            for j in range(self.bus_num):
                genName = self.IndexToGenName[i + 1]
                nodeName = self.IndexToNodeName[j]
                if genName >= nodeName:
                    com_node = G_dic[genName + nodeName]
                else:
                    com_node = G_dic[nodeName + genName]
                Z_ToSource = ToSource[com_node]
                gradientMat_p[i, j] = Z_ToSource.real * 2
                gradientMat_q[i, j] = Z_ToSource.imag * 2
                # print(1)
        return gradientMat_p, gradientMat_q

def interpolinate(demand,num=50):
    new_array=[]
    for i in range(len(demand)-1):
        new=np.linspace(demand[i],demand[i+1],num,endpoint=False)
        new_array.extend(new)
    new_array=np.asarray(new_array)
    return new_array