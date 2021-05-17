from numpy.core.defchararray import less
from graph import *
from heap import *
import numpy as np

###########################
# BINHEAP UPDATE DISTANCE #
###########################

def update_distance(H, vertice, new_distance):
    """ Function for updating Node objects' distances

    Parameters
    ----------
    H: binheap\n
        \theap storing all the Nodes of a graph
    vertice: Node\n
        \tvertice whose distance is to be updated
    new_distance: Number\n
        \tnew value for distance

    Raises
    ------
    TypeError\n
        \tif passed vertice parameter is not a Node object  
    """

    if not isinstance(vertice, Node): 
        raise TypeError('update_distance(H, vertice, new_distance): vertice must be a Node object')

    vertice.distance = new_distance
    where_vertice = vertice.heap_index
    H.decrease_key(where_vertice, vertice)

###############################
# BINHEAP VERSION OF DIJKSTRA #
###############################

def init_sssp(graph):
    """ Function for initializing graph distances and predecessors

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object to be initialized

    Raises
    ------
    TypeError\n
        \tif passed graph object is not a WeightedGraph object
        \tif vertices in the graph are not Node objects  
    """

    if not isinstance(graph, WeightedGraph):
        raise TypeError('init_sssp(graph): graph must be a WeightedGraph object')

    for key in graph.Dictionary:

        if not isinstance(graph.Dictionary[key], Node): 
            raise TypeError('init_sssp(graph): vertices are required to be Node objects')

        graph.Dictionary[key].distance = np.Inf
        graph.Dictionary[key].predecessor = None

def prepare_heap(graph, total_order):
    """ Function for building the heap containing graph vertices and assigning 
        'heap_index's to vertices

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph whose vertices must be inserted in a heap
    total_order: Function\n
        \ttotal order according to which vertices must be ordered in the heap
    """

    H = binheap(graph.Vertices, total_order=total_order)
    index = 0
    for vertice in H:
        vertice.heap_index = index
        index += 1

    return H

def relax(H, vertice1, vertice2, weight):
    """ Function for modifying vertices' members distance and predecessor

    Parameters
    ----------
    H: binheap\n
        \theap storing all the Nodes of a graph
    vertice1: Node\n
        \tremoved minimum vertice from the heap
    vertice2: Node\n
        \tvertice found in vertice1's adjacency list
    weight: Number\n
        \tweight of edge from vertice1 to vertice2

    Raises
    ------
    TypeError\n
        \tif passed vertices are not Node objects
    """

    if not isinstance(vertice1, Node): 
            raise TypeError('relax(H, vertice1, vertice2, weight): vertice1 must be a Node object')
    if not isinstance(vertice2, Node): 
            raise TypeError('relax(H, vertice1, vertice2, weight): vertice2 must be a Node object')       

    if vertice1.distance + weight < vertice2.distance:
        update_distance(H, vertice2, vertice1.distance + weight)
        vertice2.predecessor = vertice1

def dijkstra(graph, source):
    """ Function for implementing binary heap version of dijkstra algorithm

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object on which dijkstra algorith is to be applied
    source: Int\n
        \tkey of the graph's vertice to be used as source for the algorithm

    Raises
    ------
    TypeError\n
        \tif passed graph object is not a WeightedGraph object
    ValueError\n
        \tif passed source value is not a key of the graph  
    """

    if not isinstance(graph, WeightedGraph):
        raise TypeError('dijkstra_heap(graph, source): graph must be a WeightedGraph object')
    if source not in graph.Keys:
        raise ValueError('dijkstra_heap(graph, source): source must be a key of the graph but {} is not a key of the graph'.format(source))
    
    # initialize framework
    init_sssp(graph)
    graph.Dictionary[source].distance = 0

    H = prepare_heap(graph, vertice_order)

    while not H.is_empty():
        vertice_min = H.remove_min()
        for _, vertice, weight in vertice_min.adj_list:
            relax(H, vertice_min, graph.Dictionary[vertice], weight)

#############
# SHORTCUTS #
#############

def build_shortcuts(graph):
    """ Function for building shortcuts in the graph

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph in which shortcuts are wanted

    Raises
    ------
    TypeError\n
        \tif passed graph object is not a WeightedGraph
    """

    if not isinstance(graph, WeightedGraph):
        raise TypeError('build_shortcuts(graph): graph must be a WeightedGraph object')

    # check if shortcuts are already present
    if graph.shortcutted >= 1:
        return
    else:
        graph.shortcutted = 1

    # construct an heap based on importance order so vertices that between which
    # shortcuts are allowed can be identified
    H = prepare_heap(graph, importance_order)

    # for each vertice in the heap we check: minimum -> child -> granchild
    # and compare importances of these vertices. If child has an importance
    # smaller that minimum's and granchild's, then it's good for a shortcut
    # between minimum and granchild. 
    # Since child and granchild are found in adj_lists, they can have importan-
    # ce smaller than minimum's
    while not H.is_empty():

        less_important = H.remove_min()

        for _, child, weight_child in less_important.adj_list:

            if graph.Dictionary[child].importance >= less_important.importance:
                continue

            for _, granchild, weight_granchild in graph.Dictionary[child].adj_list:

                if graph.Dictionary[granchild].importance <= graph.Dictionary[child].importance:
                    continue
                if graph.Dictionary[granchild].importance == less_important.importance:
                    continue

                if less_important.shortcuts is None:
                    less_important.shortcuts = []

                new_pair = [child, granchild, weight_child + weight_granchild]
                less_important.shortcuts.append(new_pair)

def update_adj_list(vertice):
    """ Function for adding shortcuts, if they exist, to adjacency lists of vertices

    Parameters
    ----------
    vertice: Node\n
        \tvertice whose adjacency list is to be enriched by shortcuts departing from
        \tthe vertice itself  
    """

    if vertice.shortcuts is None:
        return

    for passage, shortcut, weight_shortcut in vertice.shortcuts:
        
        vertice.adj_list.append([passage, shortcut, weight_shortcut])

def update_graph(graph):
    """ Function for updating adjacency list in a graph with the existing shortcuts in
        the graph itself

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object to be updated

    Raises
    ------
    TypeError\n
        \tif passed graph object is not a WeightedGraph object 
    """

    if not isinstance(graph, WeightedGraph):
        raise TypeError('update_graph(graph): graph must be a WeightedGraph object')

    # check if graph has been already updated
    if graph.updated >= 1:
        return
    else:
        graph.updated = 1

    # update every vertice of the graph
    for vertice in graph.Vertices:
        update_adj_list(vertice)

##########################
# BIDIRECTIONAL DIJKSTRA #
##########################

def framework(graph, source1, source2):
    """ Function for setting up the framework in which the bidirectional version
        of dijkstra algorithm operates  

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object on which to apply bidirectional version of dijkstra algorithm
    source1: Int\n
        \tkey associated to the starting point of the path
    source2: Int\n
        \tkey associated to the end point of the graph

    Raises
    ------
    TypeError\n
        \tif graph is not a WeightedGraph object
    ValueError\n
        \tif passed source1 and/or source2 values do not match with any of the graph's keys
    """

    if not isinstance(graph, WeightedGraph):
        raise TypeError('bi_dijkstra(graph, start, destination): graph must be WeightedGraph object')
    if source1 not in graph.Keys:
        raise ValueError('bi_dijkstra(graph, source1, source2): source1 must be key of the graph')
    if source2 not in graph.Keys:
        raise ValueError('bi_dijkstra(graph, source1, source2): source2 must be key of the graph')

    build_shortcuts(graph)
    update_graph(graph)

    graph_up = graph.Graph_Up()

    graph_down = graph.Graph_Down()
    graph_down.Ancestors()

    init_sssp(graph_up)
    init_sssp(graph_down)

    graph_up.Dictionary[source1].distance = 0
    graph_down.Dictionary[source2].distance = 0

    H_up = prepare_heap(graph_up, vertice_order)
    H_down = prepare_heap(graph_down, vertice_order)

    return graph_up, graph_down, H_up, H_down

def bi_relax(H, vertice1, vertice2, weight, passage, graph):
    """ Bidirectional version of the relax function

    Parameters
    ----------
    H: binheap\n
        \theap storing all the Nodes of a graph
    vertice1: Node\n
        \tremoved minimum vertice from the heap
    vertice2: Node\n
        \tvertice found in vertice1's adjacency list
    weight: Number\n
        \tweight of edge from vertice1 to vertice2
    passage: Int\n
        \tkey associated to the node ignored by a shortcut between vertice1 and vertice2,
        \tif any exists.
    graph: WeightedGraph\n
        \tgraph object
    """

    if vertice1.distance + weight < vertice2.distance:

        update_distance(H, vertice2, vertice1.distance + weight)

        key_vertice1_graph = graph.Keys[graph.Vertices.index(vertice1)]
        key_vertice2_graph = graph.Keys[graph.Vertices.index(vertice2)]
        if passage is not None:
            graph.Dictionary[key_vertice2_graph].predecessor = graph.Dictionary[passage]
            graph.Dictionary[passage].predecessor = graph.Dictionary[key_vertice1_graph]
        else:
            graph.Dictionary[key_vertice2_graph].predecessor = graph.Dictionary[key_vertice1_graph]

def most_important_to_source(most_important, up=True):
    """ Function for computing a path from most important node to a source

    Parameters
    ----------
    most_important: Node\n
        \tnode from which, following predecessors, one gets to the beginning of the path
    up: bool\n
        \tboolean for chosing which path to build, if the one from the most important node
        \tto the start or to the end  
    """

    path = [most_important]

    # follow predecessors
    step = most_important.predecessor
    while step is not None:
        path = [step] + path
        step = step.predecessor

    if not up:
        return path[::-1]

    return path

def source_to_source(most_important_up, most_important_down):
    """ Function for computing the complete path from a source to another

    Parameters
    ----------
    most_important_up: Node\n
        \tmost important node in the path, contained in graph_up
    most_important_down: Node\n
        \tmost important node in the path, contained in graph_down
    """

    return most_important_to_source(most_important_up)[:-1] + most_important_to_source(most_important_down, up=False)

def one_way_path(most_important, total_distance, to_source2, to_source1):
    """ Function for returing correct path from source to most important node
        or viceversa

    Parameters
    ----------
    most_important: Node\n
        \tmost important node in the path
    total_distance: Number\n
        \tdistance computed up to when the node removed from one of the heap equals
        \tone of the sources
    to_source2, to_source1: List[Number, Node]\n
        \tlist containg the distance computed from the beginning to when the other source
        \tgets visited in the same type of graph (graph_up or graph_down)
    """

    if total_distance == min(total_distance, to_source2[0], to_source1[0]):
        return most_important_to_source(most_important), total_distance
    elif to_source2[0] == min(total_distance, to_source2[0], to_source1[0]):
        return most_important_to_source(to_source2[1]), to_source2[0]
    else:
        return most_important_to_source(to_source1[1], up=False), to_source1[0]

def path(most_important_up, most_important_down, total_distance, to_source2, to_source1):
    """ Function for returning correct path from source to source

    Parameters
    ----------
    most_important_up: Node\n
        \tmost important node in the path, contained in graph_up
    most_important_down: Node\n
        \tmost important node in the path, contained in graph_down
    total_distance: Number\n
        \tdistance computed up to when the node removed from one of the heap equals
        \tone of the sources
    to_source2, to_source1: List[Number, Node]\n
        \tlist containg the distance computed from the beginning to when the other source
        \tgets visited in the same type of graph (graph_up or graph_down)
    """

    if total_distance == min(total_distance, to_source2[0], to_source1[0]):
        return source_to_source(most_important_up, most_important_down), total_distance
    elif to_source2[0] == min(total_distance, to_source2[0], to_source1[0]):
        return most_important_to_source(to_source2[1]), to_source2[0]
    else:
        return most_important_to_source(to_source1[1], up=False), to_source1[0]  

def update_predecessors(graph, path):
    """ Function for returning graph updated with correct predecessors in the path

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object containing the path
    path: List[Node]\n
        \tpath found in the graph
    """

    for i in range(1,len(path)):
        graph.Dictionary[graph.Keys[graph.Vertices.index(path[i])]].predecessor = copy.deepcopy(path[i-1])
        

def bi_dijkstra(graph, source1, source2):
    """ Function for implementing bidirectional version of dijkstra algorithm

    Parameters
    ----------
    graph: WeightedGraph\n
        \tgraph object on which dijkstra algorith is to be applied
    source1: Int\n
        \tkey associated to the starting point of the path
    source2: Int\n
        \tkey associated to the end point of the graph
    """

    graph_up, graph_down, H_up, H_down = framework(graph, source1, source2)

    found_source2_after = [np.Inf, None]
    found_source1_after = [np.Inf, None]
    min_down = None
    while not H_up.is_empty() or H_down.is_empty():

        min_up = H_up.remove_min()

        # if newly extracted node from H_up is equal to destination,end process,
        # taking into account also previously stored distances and nodes of the 
        # destination found in some adjacent lists
        if min_up == graph.Dictionary[source2]:

            return one_way_path(min_up, min_up.distance, found_source2_after, found_source1_after)
        
        for passage, vertice, weight in min_up.adj_list:

            bi_relax(H_up, min_up, graph_up.Dictionary[vertice], weight, passage, graph_up)
    
            # if destination is in the adjacent list, take that into account storing distance and node
            if vertice == source2:
                found_source2_after = [graph_up.Dictionary[vertice].distance, copy.deepcopy(graph_up.Dictionary[vertice])]

        if min_down is not None:
            # if the newly extracted node from H_up is equal to the previously extracted node in H_down, end process
            # taking into account also previously stored distances and nodes of the destination found in some adjacent lists
            if min_up == min_down:

                return path(min_up, min_down, min_up.distance + min_down.distance, found_source2_after, found_source1_after)

        min_down = H_down.remove_min()

        # if newly extracted node from H_down is equal to destination, end process
        # taking into account also previously stored distances and nodes of the 
        # destination found in some adjacent lists
        if min_down == graph.Dictionary[source1]:

            return one_way_path(min_down, min_down.distance, found_source2_after, found_source1_after)
        
        for passage, vertice, weight in min_down.ancestors:

            bi_relax(H_down, min_down, graph_down.Dictionary[vertice], weight, passage, graph_down)
        
            # if destination is in the adjacent list, take that into account storing distance and node
            if vertice == source1:
                found_source1_after = [graph_down.Dictionary[vertice].distance, copy.deepcopy(graph_down.Dictionary[vertice])]

        # if the extracted nodes from H_up and H_down, end process taking into account also 
        # previously stored distances and nodes of the destination found in some adjacent lists
        if min_up == min_down:

            return path(min_up, min_down, min_up.distance + min_down.distance, found_source2_after, found_source1_after)