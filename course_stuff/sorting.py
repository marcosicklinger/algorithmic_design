from typing import TypeVar, List, Union, Optional, Any, Callable
# from collections.abc import Callable
from random import random
from sys import stdout
from timeit import timeit
from binheap import binheap, max_order

T = TypeVar('T')

TOrderType = Callable[[T,T], bool]

def min_order(a: T, b: T) -> bool:
    return a <= b

def di_search(A: Any, value: T, total_order: Optional[TOrderType]  = None) -> Union[None, int]:   # None: for unsuccessfull searches
    l = 0
    r = len(A)-1   # same story of indeces

    if total_order is None:
        total_order = min_order

    while r  >= l:
        m = (l+r)//2
        if total_order(A[m], value):   # A[m].id <= value
            if total_order(value, A[m]):   # A[m].id >= value
                return m
            
            l = m+1
        else:   # A[m].id > value
            r = m-1
    
    return None

def insertion_sort(A: List[T], begin: int = 0, end: Optional[int] = None, total_order: Optional[TOrderType] = None) -> None:

    if total_order is None:
        total_order = min_order

    end = end or len(A)-1 
    # ^
    # |
    # v
    # if end is None:
    #   end = len(A)-1
    #
    # this is true only if end IS NOT zero, because 0 -> False!

    for i in range(begin+1,end+1):
        j = i
        while j > begin and not total_order(A[j-1], A[j]):   # A[j] < A[j-1] <=> not (A[j-1] <= A[j])
            A[j], A[j-1] = A[j-1], A[j]   # same time operations => this is swapping
            j -= 1

def partition(A: List[T], end: int, pivot: int, begin: int = 0, total_order: Optional[TOrderType] = min_order) -> int:
    A[begin], A[pivot] = A[pivot], A[begin]

    pivot = begin
    begin = begin +1

    while end >= begin:   # while j >= i of the slides
        if total_order(A[begin],A[pivot]):
            begin += 1
        else: 
            A[begin], A[end] = A[end], A[begin]
            end -= 1

    A[pivot], A[end] = A[end], A[pivot]
    
    return end   # end is now the new index for the pivot

def select_pivot(A: List[T], begin: int, end: int, total_order) -> int:
    # base case
    if end-begin < 5: # 5 == chunk's size
        insertion_sort(A, begin=begin, end=end, total_order=total_order)
        return (begin+end)//2
    
    c_begin = begin
    pos = begin
    while c_begin+2 < end +1:
        insertion_sort(A, begin=c_begin, end=min(end,c_begin+4), total_order=total_order)
        A[pos], A[c_begin] = A[c_begin], A[pos]

        pos += 1
        c_begin += 5

    return select(A, (begin+pos-1)//2, begin=begin, end=pos-1, total_order=total_order)

def select(A: List[T], i: int, begin: int = 0, end: Optional[int] = None, total_order: Optional[TOrderType] = min_order) -> int:
    if end is None:
        end = len(A)-1
    
    # base case
    if end-begin < 5:
        insertion_sort(A, begin=begin, end=end, total_order=total_order) # quadratic, but we put a limit on size (140) -> CONSTANT from asymptotic point of view
        return i

    pivot = select_pivot(A, begin, end, total_order)
    pivot = partition(A, end, pivot, begin, total_order=total_order)

    if i == pivot:
        return i
    
    if i > pivot:
        return select(A, i, begin=pivot+1, end=end, total_order=total_order)

    return select(A, i, begin=begin, end=pivot-1, total_order=total_order)
    

def quicksort(A: List[T], begin: Optional[int] = 0, end: Optional[int] = None, total_order: Optional[TOrderType] = min_order): 
    if end is None:
        end = len(A)-1

    while begin < end:
        pivot = partition(A, end, begin, begin, total_order = total_order)

        quicksort(A, begin, pivot-1, total_order = total_order)
        # quicksort(A, pivot+1, end)
        begin = pivot+1
    
def bubble_sort(A: List[T], begin: Optional[int] = 0, end: Optional[int] = None, total_order: Optional[TOrderType] = min_order) -> None:
    if end is None:
        end = len(A)-1

    for i in range(end, begin, -1):
        for j in range(begin, i):
            if total_order(A[j], A[j+1]):
                A[j], A[j+1] = A[j+1], A[j]

def reversed_order(total_order: TOrderType) -> TOrderType:
    return (lambda a,b: total_order(b,a))

def heapsort (A: List[T], total_order: Optional[TOrderType] = min_order):
    H = binheap(A, total_order = reversed_order(total_order))   # build a max heap

    for i in range(len(A)-1, 0, -1):
        A[i] = H.remove_min()   # extract maximum from heap

def counting_sort(A: List[T]) -> List[T]:   # we need to return an array

    min_A = min(A)
    # allocate and initialize C
    C = [0]*(max(A)+1-min_A)
    # count number of repetition of each value in A
    for value in A:
        C[value-min_A] += 1
    # evaluate number of values in A smaller or equal than j
    for j in range(1, len(C)):
        C[j] += C[j-1]
    # build returning array
    B = [None]*len(A)
    # order all A's value in B
    for value in reversed(A):
        B[C[value-min_A]-1] = value
        C[value-min_A] -= 1

    return B

def bucket_sort(A: List[T]):
    # assuming uniform distribution in [0,1) for values in A

    buckets = [[] for i in range(len(A))]

    for value in A:
        idx = min(int(value*len(A)), len(buckets)-1)
        buckets[idx].append(value)

    for j in range(len(buckets)):
        insertion_sort(buckets[j])

    i = 0
    for bucket in buckets:
        for value in bucket:
            A[i] = value
            i += 1

def build_dataset(num_of_arrays: int, size: int) -> List[List[float]]:
    dataset = [None]*num_of_arrays

    for i in range(num_of_arrays):
        dataset[i] = [random() for i in range(size)]
    
    return dataset

def sort_dataset(dataset, alg):
    for A in dataset:
        alg(A)

if __name__ == '__main__':

    algorithms = ['insertion_sort', 'quicksort']
    dataset_size = 10**6

    # Header
    stdout.write('size')
    for alg in algorithms:
        stdout.write(f'\t{alg}')
    stdout.write('\n')

    # Test execution times
    for size in range(100,1100,100):
        dataset = build_dataset(dataset_size, size)

        stdout.write(f'{size}')

        for alg in algorithms:
            dataset_copy = [[value for value in A] for A in dataset]   # so that we have randomized dataset every time

            T = timeit(f'sort_dataset(dataset_copy, {alg})', globals=locals(), number=1)

            stdout.write('\t{:.6f} '.format(T/dataset_size))
            stdout.flush()
        stdout.write('\n')