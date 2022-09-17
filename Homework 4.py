# Work 1
from sys import argv
script_name, vir, st, prem = argv
t = int(vir) * int(st) + int(prem)
print("Имя скрипта:", script_name, t)

#Work 2
n = [2, 43, 23, 12, 5, 22, 34, 54, 11, 6, 27]
i = 0
new = []
new = [n[i+1] for i in range(len(n)-1) if n[i] < n[i+1]]
print(new)
#Work 3
lst = [i for i in range(20, 241) if i % 20 == 0]
print(lst)

lst = [i for i in range(20, 241) if i % 21 == 0]
print(lst)

# Work 4
lst = [3, 3, 2, 6, 1, 7, 5, 6, 9, 4, 8, 5, 4]
new = [n for n in lst if lst.count(n) == 1]
print(new)

#Work5
from functools import reduce
lst = [i for i in range(100, 1001) if i % 2 == 0]
def func(a, b):
    return a * b
print(reduce(func, lst))

#Work 6
from itertools import count
from itertools import cycle
num = int(input('Enter num: '))
i = 0
for el in count(int(num)):
    print(el)
    if el == 10:
        break


lst = [2, 4, 6, 4, 3, 7, 3, 6, 9]
i = 0
for el in cycle(lst):
    i +=1
    if i > 10:
        break
    print(el)

#Work 7

from itertools import count
from math import factorial
n = int(input('Enter the number: '))
def my_func():
    for el in count(1):
        yield factorial(el)
res = my_func()
a = 0
for i in res:
    if a < n:
        a += 1
        print(i)
    else:
        break
