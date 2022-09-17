#Work 1
from time import sleep
class TrafficLight:
    _color = {'Red': 7, 'Yellow': 2, 'Green': 5}
    def func(self, color):
        self._color = color
    def running(self):
        for key, value in self._color.items():
            sleep(value)
            print(key)
Traffic = TrafficLight()
Traffic.running()


#Work 2
class Road:
    _length = 100
    _width = 3
    def math(self, mass, sm):
        self.mass = mass
        self.sm = sm
        summ = self._length * self._width * mass * sm
        print(f'Масса асфальта, необходимая для покрытия дороги- {summ} тонн')
Roadmath = Road()
print(Roadmath.math(12, 2))

class Road:
    def func(self, length, width, mass, sm):
        self.mass = mass
        self.sm = sm
        self._length = length
        self._width = width
        summ = length * width * mass * sm
        return (f'Масса асфальта, необходимая для покрытия дороги- {summ} тонн')
Roadmath = Road()
print(Roadmath.func(10, 2, 12, 2))
#Work 3
class Worker:
    def __init__(self, name, surname, position, wage, bonus):
        self.name = name
        self.surname = surname
        self.position = position
        self._income = {'wage': wage, 'bonus': bonus}
class Position(Worker):
    def __init__(self, name, surname, position, wage, bonus):
        super().__init__(name, surname, position, wage, bonus)
    def get_full_name(self):
        return self.name + ' ' + self.surname
    def income(self):
        return self._income.get('wage') + self._income.get('bonus')

res = Position('John', 'Travolta', 'Actor', 23000, 4300)
print(res.get_full_name())
print(res.position)
print(res.income())

#Work 4
class Car:
    def __init__(self, speed, color, name, is_police):
        self.speed = speed
        self.color = color
        self.name = name
        self.is_police = is_police
    def go(self):
        return f'{self.name} started'
    def stop(self):
        return f'{self.name} stopped'
    def turnleft(self):
        return f'{self.name} turned left'
    def turnright(self):
        return f'{self.name} turned right'
    def show_speed(self):
        return f'Speed of {self.name} is {self.speed}'
class Towncar(Car):
    def __init__(self, speed, color, name, is_police):
        super().__init__(speed, color, name, is_police)
    def show_speed(self):
        print(f'Speed of {self.name} is {self.speed}')
        if self.speed > 60:
            return f'Over speed'

class Sportcar(Car):
    def __init__(self, speed, color, name, is_police):
        super().__init__(speed, color, name, is_police)

class Workcar(Car):
    def __init__(self, speed, color, name, is_police):
        super().__init__(speed, color, name, is_police)
    def show_speed(self):
        print(f'Speed of {self.name} is {self.speed}')
        if self.speed > 40:
            return f'Over speed'

class Policecar(Car):
    def __init__(self, speed, color, name, is_police):
        super().__init__(speed, color, name, is_police)
    def police_car(self):
        if self.is_police:
            print(f'{self.name} from police')
        else:
            print(f'{self.name} is not from police')

Volvo = Sportcar(70, 'black', 'Volvo', False)
Mersedes = Towncar(80, 'white', 'Mersedes', False)
Skoda = Policecar(50, 'Blue', 'Skoda', True)
BMW = Workcar(85, 'Brown', 'BMW', False)

print(BMW.show_speed())
print(Volvo.is_police)
print(Skoda.is_police)
print(Mersedes.turnleft())

#Work 5
class Stationery:
    def __init__(self, title):
        self.title = title
    def draw(self):
        return f'start rendering'
class Pen(Stationery):
    def __init__(self, title):
        super().__init__(title)
    def draw(self):
        return f'start rendering by pen'


class Pencil(Stationery):
    def __init__(self, title):
        super().__init__(title)
    def draw(self):
        return f'start rendering by pencil'

class Handle(Stationery):
    def __init__(self, title):
        super().__init__(title)
    def draw(self):
        return f'start rendering by handle'

pen = Pen('pen')
pencil = Pencil('pencil')
handle = Handle('handle')
print(pen.draw())
print(pencil.draw())
print(handle.draw())
print(pen.title)
print(pencil.title)
print(handle.title)