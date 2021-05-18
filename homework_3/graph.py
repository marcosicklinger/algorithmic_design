import copy
import numpy as np

########
# NODE #
########

class Node:
    ''' Class for representing WeightedGraph vertices

    Members
    -------
    value: Object\n
        \tthe object needed to be stored in the graph.
    distance: Number\n
        \tthe distance from a given source.
    importance: Number\n
        \tneeded to store the importance of a vertice
    predecessor: Node\n
        \tanother node, prior to the current one on a given path
    adj_list: List[List[key, weight]]\n
        \tcontaining pairs of keys and weight to adjacent nodes
    ancestors: List[List[key, weight]]\n
        \tcontaining pairs of keys and weight to incoming nodes
    heap_index: Int\n
        \tconvenient to store in order to keep up to date the position of the node in the heap
    shortcuts: List[List[key, Number]]\n
        \tcontaining the shortcuts' destinations and corresponding weights

    '''

    def __init__(self, value=None, distance=None, importance=None):
        self.value = value
        self.distance = distance
        self.importance = importance
        self.predecessor = None
        self.adj_list = None
        self.ancestors = None
        self.heap_index = None
        self.shortcuts = None

    def __eq__(self, other):
        ''' Equality criterion between two vertices: value and importance must be equal

        Parameters
        ----------
        othet: Node\n
            \tright-hand side of comparisons
        '''

        return self.value == other.value and self.importance == other.importance

    def __repr__(self):

        node = '('+str(self.importance)+', '+str(self.value)+')'

        return node

##################
# WEIGHTED GRAPH #
##################

class WeightedGraph:
    ''' Class for representing weighted graphs

    Members
    -------
    _graph_dict: Dictionary{int: Node}\n
        \tthe keys represent the nodes' importance; the values store objects representing the nodes.
    computation_of_ancestors: Int\n
        \tneeded to keep up to date the number of times the ancestors of vertices are computed so to avoid to have multiple repetitions
        in the ancestors' lists
    shortcutted: Int\n
        \tneeded to keep up to date number of times build_shortcuts function is applied to the graph, in order to prevent user to append
        \tsame shortcuts multiple times to the shortcuts' lists

    Parameters
    ----------
    graph_dict: Dictionary{int: Node}\n
        \tnodes' keys and instances with which user wants to populate the graph
    adj_dict: Dictionary{int: List[Tuple(int, Number)]}\n
        \tthe list stores pairs of nodes' keys and weight of corresponding edges with which the user wants to characterize
        \tthe graph
    key: int\n
        \tused to retrieve data corresponding to the node represented by the key

    Raises
    ------
    ValueError\n
        \tif a key given by the user is not present between the graph's keys.
    Exception\n
        \tif user wants to compute certain objects, such as the Ancestors list, but the graph is empty.
    '''

    def __init__(self, graph_dict=None, adj_dict=None):

        if graph_dict is None:
            graph_dict = {}
        
        self._graph_dict = copy.deepcopy(graph_dict)

        if adj_dict is not None:
            self.Build_Adj(adj_dict)

        self.computation_of_ancestors = 0
        self.shortcutted = 0
        self.updated = 0

    def Build_Adj(self, adj_dict):
        ''' Member function for computing adjacency lists of vertices

        Parameters
        ----------
        adj_dict: Dictionary{vertice_key: List[List[adjacent_vertice_key, weight]]}\n
            \tdictionary containing the keys of the graph's vertices whose associated values are
            \tpairs made of the corresponding adjacent vertices' key and weights of the edges

        Raises
        ------
        ValueError\n
            \tif keys of give adj_dict do not match with the graph's keys
        '''
        for key in adj_dict:

            if key not in self._graph_dict.keys():
                raise ValueError('Build_Adj(adj_list): key {} is not a key of the graph'.format(key))

            for i in range(len(adj_dict[key])):
                # add 'passage' member for shortcuts if needed
                if len(adj_dict[key][i]) < 3:
                    adj_dict[key][i] = [None] + adj_dict[key][i]
            
            for key_weight_in_adj in adj_dict[key]:
                if key_weight_in_adj[1] not in self._graph_dict.keys():
                    raise ValueError('Build_Adj(adj_list): key {} cannot be part of adjancecy list because it is not a key of the graph'.format(key_weight_in_adj))

            self._graph_dict[key].adj_list = copy.deepcopy(adj_dict[key])
    
    def Ancestors(self):
        ''' Member function for computing ancestors of a give vertice, i.e. those vertices whose edges point to the given vertice.

        Raises
        ------
        Exception\n
            \tif graph given for such computation is empty
        '''

        if len(list(self._graph_dict.keys())) == 0:
            raise Exception('the graph is empty')

        self.computation_of_ancestors += 1
        if self.computation_of_ancestors > 1:
            return

        for key in self._graph_dict:

            if len(self._graph_dict[key].adj_list) != 0:

                # the ancestor of each vertice in an adj list is the vertice whose
                # adj_list is been iterated on: 
                # if we are iterating on v1.adj_list = [v2, v3, ... , vk], then v1 is 
                # the ancestor of each vi, i = 2, ... , k, in v1.adj_list
                for pair in self._graph_dict[key].adj_list:

                    new_pair = pair[0], key, pair[2]

                    if self._graph_dict[pair[1]].ancestors is None:
                        self._graph_dict[pair[1]].ancestors = []

                    self._graph_dict[pair[1]].ancestors.append(new_pair)
        
        for key in self._graph_dict:
            if self._graph_dict[key].ancestors is None:
                self._graph_dict[key].ancestors = []

    def Graph_Up(self):
        ''' Member function for computing the sub-graph of current graph whose vertices connects 
            only with more important vertices
        '''

        graph_up = {key: Node(value=self._graph_dict[key].value, distance=self._graph_dict[key].distance, importance=self._graph_dict[key].importance) 
                    for key in self._graph_dict}

        new_adj_list = {}
        
        # for each vertice in the graph...
        for vertice in graph_up:

            new_adj_list[vertice] = []

            # ...check their adj_lists and store only those vertices whose importance is bigger
            for passage, child, weight_child in self._graph_dict[vertice].adj_list:

                if graph_up[child].importance > graph_up[vertice].importance:
                    new_adj_list[vertice].append([passage, child, weight_child])
        
        return WeightedGraph(graph_dict=graph_up, adj_dict=new_adj_list)


    def Graph_Down(self):
        ''' Member function for computing the sub-graph of current graph whose vertices connects 
            only with less important vertices
        '''

        graph_down = {key: Node(value=self._graph_dict[key].value, distance=self._graph_dict[key].distance, importance=self._graph_dict[key].importance) 
                    for key in self._graph_dict}

        new_adj_list = {}

        # for each vertice in the graph...
        for vertice in graph_down:

            new_adj_list[vertice] = []

            # ...check their adj_lists and store only those vertices whose importance is bigger
            for passage, child, weight_child in self._graph_dict[vertice].adj_list:

                if graph_down[child].importance < graph_down[vertice].importance:
                    new_adj_list[vertice].append([passage, child, weight_child])
        
        return WeightedGraph(graph_dict=graph_down, adj_dict=new_adj_list)


    @property
    def Keys(self):
        ''' Member function for returning list of all the graph's keys
        '''

        return list(self._graph_dict.keys())

    @property
    def Vertices(self):
        ''' Member function for returning list of all the graph's vertices
        '''

        return list(self._graph_dict.values())

    @property
    def Dictionary(self):
        ''' Member function for returning the graph's dictionary
        '''
        return self._graph_dict

    def __repr__(self):
        graph = 'NODES:\t--(IMPORTANCE, VALUE)--PREDECESSOR--ADJACENCY LIST\n\n'
        for item in self._graph_dict.items():
            graph += str(item[0])+':'+'\t--'+str(item[1])+'--'+str(item[1].predecessor)+'--'+str(item[1].adj_list)+'\n\n'
        return graph





