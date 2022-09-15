Work 1
with open(r'my_file.txt', 'w+', encoding='utf-8') as file:
    file.write(input('Enter the data: '))
    while True:
        data = input()
        file.write('\n')
        if data == '':
            break
        file.write(data)


Work 2
with open(r'work_2.txt', 'w+', encoding='utf-8') as w2:
    num = 89271084730
    res = w2.write(f'My dear friend!\nhow are you?\nNice to meet you!\nIm fine\nMe phone: {num}')
with open(r'work_2.txt', 'r', encoding='utf-8') as w2:
    con = w2.readlines()
    print(f'Strips: ', len(con))
with open(r'work_2.txt', 'r', encoding='utf-8') as w2:
    con2 = w2.readlines()
    for line in con2:
        l = line.split()
        print(f'Words: ', len(l))

Work3
with open(r'work_3.txt', 'w+', encoding='utf-8') as w3:
    w3.write(f'Иванов 23443\nПетров 15436\nСидоров 17433\nГрачев 16596\nСечин 24276\nТолстов 20466\nКотов 14975\nУткин 16398\nВикторов 17308\nЛеснов 25677')
    w3.seek(0)
    str = w3.read().split('\n')
    mon = []
    empl = []
    for i in str:
        i = i.split()
        if int(i[1]) < 20000:
            mon.append(i[1])
            empl.append(i[0])
    print(f'Оклад ниже 20000:{empl}, Средний заработок:{round(sum(map(int, mon)) / len(mon), 2)}')

Work 4
with open(r'work_4.txt', 'w+', encoding='utf-8') as w4:
    w4.write(f'One - 1\nTwo - 2\nThree - 3\nFour - 4')
    w4.seek(0)
    words = ['One', 'Two', 'Three', 'Four']
    numbers = [1, 2, 3, 4]
    di = dict(zip(words, numbers))
    di['Один'] = di.pop('One')
    di['Два'] = di.pop('Two')
    di['Три'] = di.pop('Three')
    di['Четыре'] = di.pop('Four')
    t = list(di.items())
    li = []
with open(r'work_4.1.txt', 'w+', encoding='utf-8') as wa:
    for a, b in di.items():
        c = str(a) + '-' + str(b)
        print(c, file=wa)
        print(c)

Work 5
with open(r'work_5.txt', 'w+', encoding='utf-8') as w5:
    w5.write(f'1 20 25 71 8 11 3 8 6 8')
    w5.seek(0)
    text = w5.read().split()
    new = sum(map(int, text))
    print(new)