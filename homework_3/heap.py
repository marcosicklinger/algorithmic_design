from graph import *

#####################
# PARAMETRIC ORDERS #
#####################

def min_order(a, b):

    return a <= b

def vertice_order(vertice1, vertice2):
    ''' Function used for minimum order type comparisons between distances of two nodes

    Parameters
    ----------
    vertice1: Node\n
        \tleft-hand side member of the comparison
    vertice2: Node\n
        \tright-hand side member of the comparison

    Raises
    ------
    TypeError\n
        \tif non-Node objects are passed as parameters
    '''
    
    if not isinstance(vertice1, Node):
        raise TypeError('vertice_order(vertice1, vertice2): vertice1 must be Node objects')
    if not isinstance(vertice2, Node):
        raise TypeError('vertice_order(vertice1, vertice2): vertice2 must be Node objects')

    return vertice1.distance < vertice2.distance

def importance_order(vertice1, vertice2):
    """  Function used for minimum order type comparisons between importances of two nodes

    Parameters
    ----------
    vertice1: Node\n
        \tleft-hand side member of the comparison
    vertice2: Node\n
        \tright-hand side member of the comparison

    Raises
    ------
    TypeError\n
        \tif non-Node objects are passed as parameters
    """

    if not isinstance(vertice1, Node):
        raise TypeError('importance_order(vertice1, vertice2): vertice1 must be Node objects')
    if not isinstance(vertice2, Node):
        raise TypeError('importance_order(vertice1, vertice2): vertice2 must be Node objects')

    return vertice1.importance < vertice2.importance

########
# HEAP #
########

class binheap:
    """ Class for representing binary heap 

    Members
    -------
    _A: List[Generic]\n
        \tstoring elements to be inserted in the heap
    _size: Int\n
        \tsize of the heap
    _torder: Function\n
        \tfunction performing comparisons according to a give total order

    Parameters
    ----------
    A: List[Generic]\n
        \tPassed list of element to be inserted in the heap
    total_order: Function\n
        \tfunction performing comparisons according to a give total order, with 
        \twhich the heap will order the nodes 
    node: Int\n
        \tindex of an element in the heap
    value: Generic\n
        \tnew item which is inserted in the heap   

    Raises
    ------
    RunTimeError\n
        \tif heap is empty and removal of minumum element is performed
        \tif heap is full and insertion is performed
        \tif bigger value is assigned to an element while trying to decrease a key
    
    Message
    -------
    The implementation of this binary heap class is the same as the one done in class, 
    with very small modifications:\n
    -- getitem method has been implemented\n
    -- _swap_keys has been suitably modified so that it can deal with the 'heap_index' member 
          of a WeightedGraph vertice (Node class). 
    """

    LEFT = 0
    RIGHT = 1

    def __init__(self, A, total_order = None):

        if total_order is None:
            self._torder = min_order
        else:
            self._torder = total_order

        if isinstance(A, int):
            self._size = 0
            self._A = [None]*A
        else:
            self._size = len(A)
            self._A = A 

        self._build_heap()

    @staticmethod
    def parent(node):

        if node == 0: 
            return None

        return (node-1)//2 

    @staticmethod 
    def child(node, side):
        
        return 2*node + 1 + side

    @staticmethod
    def left(node):
        return 2*node + 1   

    @staticmethod
    def right(node):
        return 2*node + 2
    
    def __len__(self):
        return self._size

    def _swap_keys(self, node_a, node_b):
        tmp = self._A[node_a]
        self._A[node_a] = self._A[node_b]
        self._A[node_b] = tmp

        # if we are dealing with Node objects, when swapping keys, the swapping of
        # the corresponding 'heap_index's is performed, so that the 'heap_index' of
        # a Node object in the heap always 'points' to the correct position of that
        # element in the heap 
        if isinstance(self._A[node_a], Node) and isinstance(self._A[node_b], Node):
            tmp = self._A[node_a].heap_index
            self._A[node_a].heap_index = self._A[node_b].heap_index
            self._A[node_b].heap_index = tmp

    def _heapify(self, node):

        keep_fixing = True 

        while keep_fixing:
            min_node = node
            for child_idx in [binheap.left(node), binheap.right(node)]:
                if child_idx < self._size and self._torder(self._A[child_idx], self._A[min_node]):
                    min_node = child_idx

            if min_node != node:
                self._swap_keys(min_node, node)
                node = min_node
            else: 
                keep_fixing = False

    def is_empty(self):
        return self._size == 0

    def remove_min(self):

        if self.is_empty():
            raise RuntimeError('Heap is empty')

        self._swap_keys(0,self._size - 1)

        self._size = self._size - 1

        self._heapify(0)    

        return self._A[self._size] 

    def _build_heap(self):
        for i in range(binheap.parent(self._size-1),-1,-1):     
            self._heapify(i)

    def decrease_key(self, node, new_value):
        if self._torder(self._A[node], new_value):
            raise RuntimeError(f'{new_value} is not smaller than' + f' {self._A[node]}')

        self._A[node] = new_value

        parent = binheap.parent(node)

        index = None
        while (node != 0 and not self._torder(self._A[parent], self._A[node])):
            self._swap_keys(node, parent)
            
            node = parent
            parent = binheap.parent(node)

    def insert(self, value):
        if self._size >= len(self._A):
            raise RuntimeError('heap is full')

        if self.is_empty():
            self._A[0] = value
            self._size = self._size + 1
        else:
            parent = binheap.parent(self._size)
            if self._torder(self._A[parent], value):
                self._A[self._size] = value
                self._size = self._size + 1
            else:
                self._A[self._size] = self._A[parent]
                self._size = self._size + 1
                _ = self.decrease_key(self._size - 1, value)
        
    def __getitem__(self, node):
        return self._A[node]

    def __repr__(self):
        
        bh_str = ''

        next_node = 1
        up_to = 2

        while next_node <= self._size:
            level = '\t'.join(f'{v}' for v in self._A[next_node-1: min(up_to-1,self._size)]) 

            if next_node == 1:
                bh_str = level
            else: 
                bh_str += f'\n{level}'

            next_node = up_to
            up_to = 2*up_to

        return bh_str