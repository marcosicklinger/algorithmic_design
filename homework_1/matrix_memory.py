from __future__ import annotations

from numbers import Number
from typing import List
from typing import Tuple


def gauss_matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    ''' Multiply two matrices by using Gauss's algorithm

    Parameters
    ----------
    A: Matrix
        The first matrix to be multiplied
    B: Matrix
        The second matrix to be multiplied

    Returns
    -------
    Matrix
        The row-column multiplication of the matrices passed as parameters

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''

    if A.num_of_cols != B.num_of_rows:
        raise ValueError('The two matrices cannot be multiplied')

    result = [[0 for col in range(B.num_of_cols)] for row in range(A.num_of_rows)]

    for row in range(A.num_of_rows):
        for col in range(B.num_of_cols):
            value = 0
            for k in range(A.num_of_cols):
                value += A[row][k] * B[k][col]

            result[row][col] = value

    return Matrix(result, clone_matrix=False)

def get_matrix_quadrants(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    ''' Return the matrix quadrants

    Parameters
    ----------
    A: Matrix
        The matrix of which the quadrants are requested

    Returns
    -------
    Tuple(Matrix, Matrix, Matrix, Matrix)
        Tuple made of the quadrants of the passed matrix
    '''

    A11 = A.submatrix(0 , A.num_of_rows//2, 0, A.num_of_cols//2)
    A12 = A.submatrix(0 , A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)
    A21 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, 0, A.num_of_cols//2)
    A22 = A.submatrix(A.num_of_rows//2, A.num_of_rows//2, A.num_of_cols//2, A.num_of_cols//2)

    return A11, A12, A21, A22

def get_matrix_quadrant(A: Matrix, i: int, j: int) -> Matrix:
    ''' Return selected matrix quadrant

    Parameters
    ----------
    A: Matrix
        The matrix of which a specific quadrant is requested
    i: Integer
        First coordinate of the desired quadrant
    j: Integer
        Second coordinate of the desired quadrant 

    Returns
    -------
    Matrix
        The quadrant corresponding to the Aij sub-matrix of the matrix A

    Raises
    ------
    ValueError
        If the pair i,j does not correspond to one of the four matrix quadrants
    '''

    if i<1 or i>2 or j<1 or j>2:
        raise ValueError('unidentified quadrant')

    return A.submatrix((i-1)*A.num_of_rows//2,A.num_of_rows//2, (j-1)*A.num_of_cols//2, A.num_of_cols//2)

def S_matrix(A: Matrix, i: int, j: int, l: int, m: int) -> Matrix:
    ''' Return S matrix needed for the Strassen's algorithm

    Parameters
    ----------
    A: Matrix
        The matrix from which to compute the corresponding S matrix
    i: Integer
        First coordinate of the quadrant that will be used as first operand 
    j: Integer
        Second coordinate of the quadrant that will be used as first operand
    l: Integer
        First coordinate of the quadrant that will be used as second operand
    m: Integer
        Second coordinate of the quadrant that will be used as second operand

    Returns
    -------
    Matrix
        The S matrix corresponding to the computation Aij +- Alm 
    '''

    if i==l or i==j and l==m:
        return get_matrix_quadrant(A,i,j) + get_matrix_quadrant(A,l,m)
    else:
        return get_matrix_quadrant(A,i,j) - get_matrix_quadrant(A,l,m)

def strassen_matrix_mult(A: Matrix, B: Matrix) -> Matrix:
    ''' Multiply two matrices by using Strassen's algorithm

    Parameters
    ----------
    A: Matrix
        The first matrix to be multiplied
    B: Matrix
        The second matrix to be multiplied

    Returns
    -------
    Matrix
        The matrix computed following the Strassen's algorithm procedure
    '''

    original_nrows_A = A.num_of_rows
    original_ncols_B = B.num_of_cols

    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 64:
        return gauss_matrix_mult(A,B)

    if A.num_of_rows%2 != 0:
        A.append_null_row()

    if A.num_of_cols%2 != 0:
        A.append_null_column()
        B.append_null_row()

    if B.num_of_cols%2 != 0:
        B.append_null_column()

    result = Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)], clone_matrix=False)

    result.assign_submatrix(0, 0, strassen_matrix_mult(S_matrix(A,1,1,2,2), S_matrix(B,1,1,2,2)) 
                                  + strassen_matrix_mult(get_matrix_quadrant(A,2,2), S_matrix(B,2,1,1,1)) 
                                  - strassen_matrix_mult(S_matrix(A,1,1,1,2), get_matrix_quadrant(B,2,2))
                                  + strassen_matrix_mult(S_matrix(A,1,2,2,2), S_matrix(B,2,1,2,2)))
    result.assign_submatrix(0, result.num_of_cols//2, strassen_matrix_mult(get_matrix_quadrant(A,1,1), S_matrix(B,1,2,2,2))
                                                      + strassen_matrix_mult(S_matrix(A,1,1,1,2), get_matrix_quadrant(B,2,2)))
    result.assign_submatrix(result.num_of_rows//2, 0, strassen_matrix_mult(S_matrix(A,2,1,2,2), get_matrix_quadrant(B,1,1))
                                                      + strassen_matrix_mult(get_matrix_quadrant(A,2,2), S_matrix(B,2,1,1,1)))
    result.assign_submatrix(result.num_of_rows//2, result.num_of_cols//2, strassen_matrix_mult(S_matrix(A,1,1,2,2), S_matrix(B,1,1,2,2))
                                                                          + strassen_matrix_mult(get_matrix_quadrant(A,1,1), S_matrix(B,1,2,2,2))
                                                                          - strassen_matrix_mult(S_matrix(A,2,1,2,2), get_matrix_quadrant(B,1,1))
                                                                          - strassen_matrix_mult(S_matrix(A,1,1,2,1), S_matrix(B,1,1,1,2)))

    result = result.submatrix(0, original_nrows_A, 0, original_ncols_B)

    return result

class Matrix(object):
    ''' A simple naive matrix class

    Members
    -------
    _A: List[List[Number]]
        The list of rows that store all the matrix values

    Parameters
    ----------
    A: List[List[Number]]
        The list of rows that store all the matrix values
    clone_matrix: Optional[bool]
        A flag to require a full copy of `A`'s data structure.

    Raises
    ------
    ValueError
        If there are two lists having a different number of values
    '''
    def __init__(self, A: List[List[Number]], clone_matrix: bool = True):
        num_of_cols = None

        for _, row in enumerate(A):
            if num_of_cols is not None:
                if num_of_cols != len(row):
                    raise ValueError('This is not a matrix')
            else:
                num_of_cols = len(row)

        if clone_matrix:
            self._A = [[value for value in row] for row in A]
        else:
            self._A = A

    @property
    def num_of_rows(self) -> int:
        return len(self._A)

    @property
    def num_of_cols(self) -> int:
        if len(self._A) == 0:
            return 0

        return len(self._A[0])

    @property
    def max(self):
        ''' Return larger element in the matrix

        Returns
        -------
        Number
            The larger element in the matrix
        '''
       
        return max(max(self._A))

    def copy(self):
        A = [[value for value in row] for row in self._A]

        return Matrix(A, clone_matrix=False)

    def __getitem__(self, y: int):
        ''' Return one of the rows

        Parameters
        ----------
        y: int
            the index of the rows to be returned

        Returns
        -------
        List[Number]
            The `y`-th row of the matrix
        '''
        return self._A[y]

    def __iadd__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] += A[y][x]

        return self

    def __add__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res += A

        return res

    def __isub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] -= A[y][x]

        return self

    def __sub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res -= A

        return res

    def __mul__(self, A: Matrix) -> Matrix:
        ''' Multiply one matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix which multiplies this matrix

        Returns
        -------
        Matrix
            The row-column multiplication between this matrix and that passed
            as parameter

        Raises
        ------
        ValueError
            If the number of columns of this matrix is different from the
            number of rows of `A`
        '''
        return gauss_matrix_mult(self, A)

    def __rmul__(self, value: Number) -> Matrix:
        ''' Multiply one matrix by a numeric value

        Parameters
        ----------
        value: Number
            The numeric value which multiplies this matrix

        Returns
        -------
        Matrix
            The multiplication between `value` and this matrix

        Raises
        ------
        ValueError
            If `value` is not a number
        '''

        if not isinstance(value, Number):
            raise ValueError('{} is not a number'.format(value))

        return Matrix([[value*elem for elem in row] for row in self._A],
                      clone_matrix=False)

    def submatrix(self, from_row: int, num_of_rows: int,
                  from_col: int, num_of_cols: int) -> Matrix:
        ''' Return a submatrix of this matrix

        Parameters
        ----------
        from_row: int
            The first row to be included in the submatrix to be returned
        num_of_rows: int
            The number of rows to be included in the submatrix to be returned
        from_col: int
            The first col to be included in the submatrix to be returned
        num_of_cols: int
            The number of cols to be included in the submatrix to be returned

        Returns
        -------
        Matrix
            A submatrix of this matrix
        '''
        A = [row[from_col:from_col+num_of_cols] for row in self._A[from_row:from_row+num_of_rows]]

        return Matrix(A, clone_matrix=False)

    def assign_submatrix(self, from_row: int, from_col: int, A: Matrix):
        for y, row in enumerate(A):
            self_row = self[y + from_row]
            for x, value in enumerate(row):
                self_row[x + from_col] = value

    def append_null_row(self):
        ''' append new row full of zeros to the matrix
        '''

        self._A.append([0 for x in range(self.num_of_cols)])

    def append_null_column(self):
        ''' append new column full of zeros to the matrix
        '''
        
        for row in self._A:
            row.append(0)

    def __repr__(self):
        return '\n'.join('{}'.format(row) for row in self._A)


class IdentityMatrix(Matrix):
    ''' A class for identity matrices

    Parameters
    ----------
    size: int
        The size of the identity matrix
    '''
    def __init__(self, size: int):
        A = [[1 if x == y else 0 for x in range(size)]
             for y in range(size)]

        super().__init__(A)

#__________________________________
#__________EXECUTION TIME__________
#__________________________________

if __name__ == '__main__':

    from random import random
    from random import seed

    from sys import stdout

    from timeit import timeit

    seed(0)

    f = open("memory.txt","w")

    for i in range(8):
        size = 2**i
        stdout.write(f'{size}')
        f.write(f'{size}')
        A = Matrix([[random() for x in range(size)] for y in range(size)])
        B = Matrix([[random() for x in range(size)] for y in range(size)])

        for funct in ['gauss_matrix_mult', 'strassen_matrix_mult']:
            T = timeit(f'{funct}(A,B)', globals=locals(), number=1) 
            stdout.write('\t{:.3f}'.format(T)) 
            stdout.flush() 
            f.write('\t{:.3f}'.format(T))

        stdout.write('\n')
        f.write('\n')
    
    f.close()