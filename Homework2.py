#Work 1
# li = [1, 3, 'good day', {'winter': 'february', 'summer': 'july', 'spring': 'april'}, [1, 3, 5, 7], True]
# for t in li:
#     print(type(t))

# #Work 2
# l = input("Введите элементы списка: ").split()
# l[:-1:2], l[1::2] = l[1::2], l[:-1:2]
# print(l)

#Work 3- через list
# t = int(input('Введите месяц от 1 до 12: '))
# l1 = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# if t in l1[:3]:
#     print('Зима')
# elif t in l1[3:6]:
#     print('Весна')
# elif t in l1[6:9]:
#     print('Лето')
# else: print('Осень')

#Work 3- через dict
# t = int(input('Введите месяц от 1 до 12: '))
# d = {'Зима': [12, 1, 2], 'Весна': [3, 4, 5], 'Лето': [6, 7, 8], 'Осень': [9, 10, 11]}
# for month in d:
#     if t in d[month]:
#         print(month)
#         break

#Work 4
# sar = input('Введите данные: ').lower().split()
# for i, el in enumerate(sar, 1):
#     print(f' {i}. {el[0:10]}')

#Work 5
# rating = [7, 5, 3, 3, 2]
# el = int(input('Enter the number: '))
# for a in rating:
#     if el > int(rating[0]):
#         rating.insert(0, el)
#         print(rating)
#         break
#     else:
#         rating.append(el)
#         rating.sort(reverse=True)
#         print(rating)
#         break