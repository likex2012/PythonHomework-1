#Work1
a = int(input('enter num1: '))
b = int(input('enter num2: '))
def f1(a, b):
    try:
        a / b
        return round(a / b, 3)
    except ZeroDivisionError:
        return 'на ноль делить нельзя'
s = f1(a, b)
print(s)

# Work2
def func(name, surname, city, email, tel):
    print(f'{name} {surname}, {city}, {email}, {tel}')
func(name= input('Enter the name: '), surname= input('Enter the surname: '), city= input('Enter the city: '), email= input('Enter email: '), tel= input('Enter the tel: '))

#Work3
a = int(input('Enter a: '))
b = int(input('Enter b: '))
c = int(input('Enter c: '))
def my_func(a, b, c):
    if a < b and a < c:
        print(b + c)
        return
    elif b < a and b < c:
        print(a + c)
        return
    else:
        print(a + b)
        return
my_func(a, b, c)

#Work4
a = int(input('Enter a: '))
b = int(input('Enter b: '))
def fun(a, b):
    while a < 0 or b > 0:
        print('error')
        if a < 0:
            a = int(input('Enter a: '))
        if b > 0:
            b = int(input('Enter b: '))
        else:
            break
    k = a ** b
    print(k)
fun(a, b)


#Work5
def fun1():
    s = 0
    while True:
        try:
            li = input('enter data: ').split()
            for n in li:
                if n == 'stop':
                    print(s)
                    return
                else:
                    s += int(n)
            print(s)
        except ValueError:
            return s
fun1()

#Work6

def int_func():
    while True:
        a = input('Enter words: ').lower().split()
        for n in a:
            if n == 'stop':
                return
            if n == n.lower():
                n.capitalize()
                print(n.capitalize(), end=(' '))

int_func()