from typing import TypeVar, Generic, Union, List
from numbers import Number

T = TypeVar('T')

# parametric orders
def min_order(a: Number, b: Number) -> bool:
    return a <= b

def max_order(a: Number, b: Number) -> bool:
    return a >= b

# class binary heap
class binheap(Generic[T]):

    LEFT = 0
    RIGHT = 1

    def __init__(self, A: Union[int, List[T]], total_order = None):

        if total_order is None:
            self._torder = min_order
        else:
            self._torder = total_order

        # if A arg is an int I want to build a heap with size equal to A
        if isinstance(A, int):
            # set the value of current valid "places" to 0 
            self._size = 0
            # but the heap itself will be able to store at most A "places"
            self._A = [None]*A
        else:
            self._size = len(A) # size of binary heap
            self._A = A # where to store the keys

        self._build_heap()

    @staticmethod # doesnt rely on value/content of the current object (binheap obj)
    def parent(node: int) -> Union[int, None]:

        if node == 0: # if i is root the parent of root doesnt exist
            return None

        return (node-1)//2 # in python indexes start from 0 => we need to reduce by one 

    @staticmethod # doesnt rely on value/content of the current object (binheap obj), taht is the method doesnt depend on any istance of the class
    def child(self, node: int, side: int) -> int:
        
        return 2*node + 1 + side

    @staticmethod
    def left(node: int) -> int:
        return 2*node + 1   # defferent from the slides, due to the fact that indexes in slides start from one, in py from 0

    @staticmethod
    def right(node: int) -> int:
        return 2*node + 2
    
    def __len__(self):
        return self._size

    def _swap_keys(self, node_a: int, node_b: int) -> None:
        tmp = self._A[node_a]
        self._A[node_a] = self._A[node_b]
        self._A[node_b] = tmp

    def _heapify(self, node: int) -> None:

        keep_fixing = True # decide if _heapify should keep going

        # while keep_fixing=True, check if root contains the minimum among the keys of the root itself and the children's. If this is the case, we dont do anything and set keep_fixing=False
        # otherwise (exist child with key smaller than root's), swap root's key and child's key, and keep doing this for every subtree of the node
        while keep_fixing:
            min_node = node
            for child_idx in [binheap.left(node), binheap.right(node)]:
                if child_idx < self._size and self._torder(self._A[child_idx], self._A[min_node]):
                    min_node = child_idx

        # min_node is the index of the minimum key among the keys of root and its children

            if min_node != node:
                self._swap_keys(min_node, node)
                node = min_node
            else: 
                keep_fixing = False

    def is_empty(self) -> bool:
        return self._size == 0

    def remove_min(self) -> T:

        if self.is_empty():
            raise RuntimeError('Heap is empty')

        self._swap_keys(0,self._size - 1)

        # self._A[0] = self._A[self._size - 1]  # copy the content of rightmost node on the last level in the root, doing that losing the value of the root
                                                # could be nice preserving it, so use the _swap_keys() function

        self._size = self._size - 1

        self._heapify(0)    # fix heap property from the root below

        return self._A[self._size] # we want to RETURN THE MINIMUM

    # method to FIX HEAP PROPERTY on the internal representetion of the array (fix heap prop bottom->up)
    def _build_heap(self) -> None:
        for i in range(binheap.parent(self._size-1),-1,-1):     # calling binheap on leaves is useless
            self._heapify(i)

    def decrease_key(self, node: int, new_value: T) -> None:
        if self._torder(self._A[node], new_value):
            raise RuntimeError(f'{new_value} is not smaller than' + f' {self._A[node]}')

        self._A[node] = new_value

        parent = binheap.parent(node)
        while (node != 0 and not self._torder(self._A[parent], self._A[node])):
            self._swap_keys(node, parent)
            parent = binheap.parent(node)

    def insert(self, value: T) -> None:
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
                self.decrease_key(self._size - 1, value)

    def __repr__(self) -> str:
        # print level by level the keys of the nodes (simple but not smart representation)
        bh_str = ''

        next_node = 1
        up_to = 2

        while next_node <= self._size:
            level = '\t'.join(f'{v}' for v in self._A[next_node-1: min(up_to-1,self._size)]) # '-1' needed because python indeces start from 0

            if next_node == 1:
                bh_str = level
            else: 
                bh_str += f'\n{level}'

            next_node = up_to
            up_to = 2*up_to

        return bh_str


