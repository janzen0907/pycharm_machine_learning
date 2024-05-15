class Car:
    '''A simple representation of a car'''

    def __init__(self, make, model, year):
        '''Initialize a new car'''
        self.make = make
        self.model = model
        self.year = year
        self._miles = 0

    def get_descriptive_name(self):
        '''Returns a descriptive name for the car'''
        descriptive_name = f'{self.year} {self.make} {self.model}'
        return descriptive_name

    def read_odometer(self):
        '''Prints the odometer's reading'''
        print(f'This car has {self._miles} miles on it')

    def update_odometer(self, milage):
        '''Updating the odometer'''
        if milage > self._miles:
            self._miles = milage
        else:
            print("YOu cant roll back an odometer")

    def __str__(self):
        '''Return a string representation of the car'''
        return f'{self.get_descriptive_name()} ({self._miles})'

    def __call__(self):
        return f"{self.get_descriptive_name()} and i have {self._miles} miles on me"

    # Most "automatic " things in Python are relying on dunder or "magic" functions.
    # If you want to override equals, or less than, length, etc you're probably
    # just changing dunder functions.


# Note you can do multiple inheritance

# Let's subclass car with an electric car
class ElectricCar(Car):
    '''Represents electric cars'''

    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery = self.Battery(40)

    def describe_battery(self):
        print(f'this car has a {self.battery.size} kwh battery')

    def read_odometer(self):
        kms = self._miles * 1.609344
        print(f'this car has {kms} km on it')

    class Battery:
        def __init__(self, size):
            self.size = size
            self.charge = 100.0

        def display_charge(self):
            print(f'The battery is at {self.charge}')


impala = Car("Chevrolt", "Impala", 2013)
print(impala.get_descriptive_name())
impala.update_odometer(3000)
impala.read_odometer()
impala.update_odometer(200)
print(impala)
impala()
leaf = ElectricCar("Nissan", 'Leaf', 2023)
leaf.update_odometer(20000)
print(leaf)
leaf.describe_battery()
leaf.read_odometer()
leaf.battery.display_charge()
