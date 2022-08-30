#Work 1
Name = input('Enter your name: ')
Age = input('Enter your age: ')
Living_city = input('Enter your living city: ')
print('Name: ', Name)
print('Age: ', Age)
print('Living city: ', Living_city)

#Work 2
time = int(input('Enter the time in sec: '))
mins = time / 60
hours = mins // 60
hours1 = hours * 60 * 60
sec1 = time - hours1
mins1 = sec1 // 60
sec2 = sec1 % 60
print(f'{int(hours)} : {int(mins1)} : {int(sec2)}')

#Work 3
n1 = input('Enter the number: ')
n2 = n1 + n1
n3 = (n2 + n1)
N = (int(n1) + int(n2) + int(n3))
print(N)

#Work 4
num = int(input('Enter the number: '))
i = str(num)
k = list(map(int, i))
m = max(k)
print(m)

#Work5
r = int(input('What is revenue?'))
c = int(input('What is the costs?'))
p = r - c
if r > c:
    print("Profit: ", p)
    profitability = p / r
    print('Profitability is: ', round(profitability, 2))
    emp = int(input('What is your number of employees?: '))
    profit_per_person = p / emp
    print('Profit per person: ', round(profit_per_person, 2))
elif r == c:
    print('Profit', p)
else:
    print('Losses', p)

#Work 6
a = 3
b = 7
i = 1
while a <= b:
    a = a + a / 10
    i = i + 1
    print(i, '-й день: ', round(a, 2))