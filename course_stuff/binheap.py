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

        self._torder = total_order
        self._size = 0 # size of binary heap
        self._A = [] # where to store the keys

    @staticmethod # doesnt rely on value/content of the current object (binheap obj)
    def parent(self, node: int) -> Union[int, None]:

        if node == 0: # if i is root the parent of root doesnt exist
            return None

        return (node-1)//2 # in python indexes start from 0 => we need to reduce by one 

    @staticmethod # doesnt rely on value/content of the current object (binheap obj)
    def child(self, node: int, side: int) -> int:
        
        return 2*node + 1 + side
    
    def __len__(self):

        return self._size

    def _swap_keys(self, node_a: int, node_b: int) -> None:
        tmp = self._A[node_a]
        self._A[node_a] = self._A[node_b]
        self._A[node_b] = tmp

    def _heapify(self, node: int) -> None:

        keep_fixing = True # decide if _heapify should keep going

        while keep_fixing:
            min_node = node
            for side in [binheap.RIGHT, binheap.LEFT]:
                child_idx = binheap.child(node, side)
                if child_idx < self._size and self._torder(self._A[child_idx], self._A[min_node]):
                    min_node = child_idx

            if min_node != node:
                self._swap_keys(min_node, node)
                node = min_node
            else: 
                keep_fixing = False


