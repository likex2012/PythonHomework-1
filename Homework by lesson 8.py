#Work 1
class Data:
    def __init__(self, data):
        self.d = data

    @classmethod
    def new(cls, data):
        my_date = []

        for el in data.split():
            if el != '-': my_date.append(el)
        return int(my_date[0]), int(my_date[1]), int(my_date[2])

    @staticmethod
    def inform(day, month, year):
        if 1 <= day <= 31:
            if 1 <= month <= 12:
                if 2022 >= year >= 0:
                    return f'All right'
                else:
                    return f'wrong year'
            else:
                return f'wrong month'
        else:
            return f'wrong day'

def __str__(self):
    return f'Current date: {Data.new(self.data)}'


today = Data('11 - 1 - 2001')
print(today)
print(Data.inform(23, 6, 2022))
print(today.inform(21, 12, 2025))
print(Data.new('16 - 10 - 2015'))
print(today.new('16 - 08 - 2021'))
print(Data.inform(13, 10, 2010))

#Work2
class New:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @staticmethod
    def new1(a, b):
        try:
            return (a / b)
        except:
            return (f"Делить на ноль нельзя")


c = New(20, 100)
print(New.new1(10, 0))
print(New.new1(10, 5))
print(c.new1(100, 0))

#Work 3
class Error:
    def __init__(self, *args):
        self.my_list = []

    def my_input(self):
        while True:
            try:
                data = int(input('Введите данные и нажмите Enter: '))
                self.my_list.append(data)
                print(f'Текущий список - {self.my_list} \n ')
            except:
                print(f'Error')
                y_or_n = input(f'Попробовать еще раз? Y/N ')

                if y_or_n == 'Y' or y_or_n == 'y':
                    print(try_except.my_input())
                elif y_or_n == 'N' or y_or_n == 'n':
                    return f'Вы вышли'
                else:
                    return f'Вы вышли'

try_except = Error(1)
print(try_except.my_input())

#Work4
class Storage:

    def __init__(self, name, price, quantity, number_of_lists, *args):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.numb = number_of_lists
        self.my_store_full = []
        self.my_store = []
        self.my_unit = {'Модель устройства': self.name, 'Цена за шт.': self.price, 'Количество': self.quantity}

    def __str__(self):
        return f'{self.name} цена {self.price} количество {self.quantity}'
    def reception(self):
        try:
            unit = input(f'Введите наименование ')
            unit_p = int(input(f'Введите цену за ед '))
            unit_q = int(input(f'Введите количество '))
            unique = {'Модель устройства': unit, 'Цена за ед': unit_p, 'Количество': unit_q}
            self.my_unit.update(unique)
            self.my_store.append(self.my_unit)
            print(f'Текущий список -\n {self.my_store}')
        except:
            return f'Error'

        print(f'Для выхода - Q, продолжение - Enter')
        q = input(f' ')
        if q == 'Q' or q == 'q':
            self.my_store_full.append(self.my_store)
            print(f'Весь склад -\n {self.my_store_full}')
            return f'Выход'
        else:
            return Storage.reception(self)
class Printer(Storage):
    def to_print(self):
        return f'to print smth {self.numb} times'


class Scanner(Storage):
    def to_scan(self):
        return f'to scan smth {self.numb} times'


class Copier(Storage):
    def to_copier(self):
        return f'to copier smth  {self.numb} times'


unit_1 = Printer('hp', 4000, 12, 4)
unit_2 = Scanner('Canon', 1000, 23, 14)
unit_3 = Copier('Xerox', 2300, 23, 5)
print(unit_1.reception())
print(unit_2.reception())
print(unit_3.reception())
print(unit_1.to_print())
print(unit_3.to_copier())