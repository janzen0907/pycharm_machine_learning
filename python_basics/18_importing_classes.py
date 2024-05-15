from cars import ElectricCar, Car

tesla = ElectricCar("Tesla", "Model S", 2023)
print(tesla)

accord = Car("Honda", "Accord", 2001)
print(accord)

from cars import Car as MyCarClass
skylark = MyCarClass("Buick", 'Skylark', 1987)
print(skylark)

import cars as cars
regal = cars.Car("Buick", "Regal", 2004)
print(regal)

print(type(regal))
print(isinstance(tesla, ElectricCar))
print(isinstance(tesla, Car))
print(isinstance(tesla, MyCarClass))
print(isinstance(skylark, ElectricCar))

tesla.update_odometer(4000)
new_tesla = tesla
print(tesla)
print(new_tesla)
# This will affect tesla as well. They are the same object.
new_tesla.update_odometer(5000)
print(tesla)
print(new_tesla)

# If you want to clone/ copy instead copying the reference,
# you can use the copy library
import copy as copy
# Can use copy for a shallow copy
# or deepcopy for a deep, recursive copy
new_new_tesla = copy.deepcopy(new_tesla)
new_new_tesla._miles = 0
print(new_tesla)
print(new_new_tesla)
new_new_tesla.battery.size = 50
print(new_tesla.battery.size)
print(new_new_tesla.battery.size)

from random import randint as ran
print(ran(1,100))

import random as ran
cars = [tesla, skylark, accord, regal]
print(f'Today we will drive the {ran.choice(cars)}')