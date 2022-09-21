# Work 1

import copy

class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __str__(self):
        string = ''
        for el in range(len(self.matrix)):
            string = string + '\t'.join(map(str, self.matrix[el])) + '\n'
        return string

    def __add__(self, other):
        if len(self.matrix) != len(other.matrix):
            return 'Error'
        result = copy.copy(self.matrix)
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                result[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return Matrix(result)
l1 = [[2, 7, 2], [4, 9, 1], [8, 2, 4]]
l2 = [[5, 3, 8], [1, 6, 4], [8, 3, 2]]
l3 = [[1, 6, 5], [3, 2, 5], [1, 2, 6], [3, 4, 6]]
a = Matrix(l1)
b = Matrix(l2)
c = Matrix(l3)
print(a)
summ = a + b
summ1 = a + c
print(c)
print(summ)
print(summ1)

# Work 2
from abc import ABC, abstractmethod
class Clothes(ABC):
    def __init__(self, data):
        self.data = data
    @abstractmethod
    def exp(self):
        pass
    def __str__(self):
        return str(self.data)
class Coat(Clothes):
    @property
    def exp(self):
        return round(self.data / 6.5 + 0.5, 2)
class Costume(Clothes):
    @property
    def exp(self):
        return round(self.data * 2 + 0.3, 2)

a = Coat(47)
b = Costume(1.65)
print(a.exp)
print(b.exp)

#Work 3
class Cell:
    def __init__(self, cell):
        self.cell = cell
        self.symbol = '*'

    def __str__(self):
        return str(f'Количество ячеек - {self.cell}')

    def __sub__(self, other):
        return Cell(abs(self.cell - other.cell))

    def __mul__(self, other):
        return Cell(self.cell * other.cell)

    def __truediv__(self, other):
        return Cell(self.cell // other.cell)

    def __add__(self, other):
        return Cell(abs(self.cell + other.cell))

    def make_order(self, count):
        x = self.cell
        while x > 0:
            for el in range(1,count+1):
                print(self.symbol, end ='')
                x -= 1
                if x <= 0:
                    break
            print('\n', end = '')



a = Cell(9)
b = Cell(12)
c = Cell(8)
d = Cell(3)

print(a + b)
print(a - b)
print(a * b)
print(c / d)

a.make_order(4)
b.make_order(3)