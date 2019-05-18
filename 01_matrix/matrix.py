import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I
#点积
def dot_product(vector_one, vector_two):
    result = 0
    for i in range(len(vector_one)):
        result += vector_one[i]*vector_two[i]
    return result
    
class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)#rows
        self.w = len(grid[0])#columns
       

    #
    # Primary matrix math methods
    #############################
 
    #计算矩阵行列式
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        # TODO - your code here
        if self.h == 1:
            return self.g[0][0]
        elif self.h == 2:
            return self.g[0][0]*self.g[1][1] - self.g[1][0]*self.g[0][1]
        
    #矩阵轨迹,轨迹是跨主对角线的和.    
    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        sum = 0;
        for i in range(self.w):
            sum += self.g[i][i]
        return sum
    
    #矩阵的逆矩阵
    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")
        inverseM = []
        # TODO - your code here
        matrix = self.g;
        if len(matrix) == 1:
            row = [1/matrix[0][0]]
            inverseM.append(row)
        elif len(matrix) == 2 and (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]) != 0:
#             row1 = []
#             row2 = []
#             val = 1/(matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0])
#             row1.append(val*matrix[1][1])
#             row1.append(-val*matrix[0][1])
#             row2.append(-val*matrix[1][0])
#             row2.append(val*matrix[0][0])
#             inverseM.append(row1)
#             inverseM.append(row2)  
            #求逆矩阵公式
            return 1.0 / self.determinant() * (self.trace() * identity(self.h) - self)
        return Matrix(inverseM)    

    
    #转置矩阵    
    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        # TODO - your code here
#         matrixT = []
#         for colIdx in range(self.w):
#             row = []
#             for rowIdx in range(self.h):
#                 row.append(self.g[rowIdx][colIdx])
#             matrixT.append(row)
#         return Matrix(matrixT)
        return Matrix(list(zip(*self.g)))#是一个把二维的数组 的行 和列交换的操作
    
    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    #矩阵加法
    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        #   
        # TODO - your code here
        #
        
#         newMatrix = []
#         for i in range(self.h):
#             row = []
#             for j in range(self.w):
#                 row.append(self.g[i][j] + other.g[i][j])
#             newMatrix.append(row) 
#         return Matrix(newMatrix)    
        return Matrix([[self[i][j] + other[i][j] for j in range(self.w)] for i in range(self.h)]) 

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #   
        # TODO - your code here
        #
        
#         newMatrix = []
#         for i in range(self.h):
#             row = []
#             for j in range(self.w):
# #                 self.g[i][j] = -self.g[i][j]
#                 row.append(-self.g[i][j])
#             newMatrix.append(row)
#         return Matrix(newMatrix)   
        return Matrix([[-self[i][j] for j in range(self.w)] for i in range(self.h)])

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #   
        # TODO - your code here
        #
#         newMatrix = []
#         for i in range(self.w):
#             row = []
#             for j in range(self.h):
#                 row.append(self.g[i][j] - other[i][j])
#             newMatrix.append(row)    
#         return Matrix(newMatrix)
        return Matrix([[self.g[i][j] - other[i][j] for j in range(self.w)] for i in range(self.h)])

    #矩阵乘法
    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #   
        # TODO - your code here
        #
        newMatrix = []
        otherT = other.T()
        for i in range(self.h):
            row = []
            for j in range(len(otherT.g)):
                row.append(dot_product(self.g[i], otherT.g[j]))
            newMatrix.append(row)

        return Matrix(newMatrix)

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            #   
            # TODO - your code here
            #
            newMatrix = []
            for i in range(self.w):
                row = []
                for j in range(self.h):
                    row.append(self.g[i][j]*other) 
                newMatrix.append(row)    
            return Matrix(newMatrix)
        
#Jiangchun
#chiangchuna@gmail.com
#May 18th, 2019  20:30        
        