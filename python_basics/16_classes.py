# Class names should start with caps
class Dog:
    '''A simple representation of a dog'''

    total_tricks_done = 0

    # Dunder/ magic metho
    # THis is called when you crreate a new instanece of the class
    # self will get passed in by python and must come first
    # self gets created bu the __new__ duncer function, but usually wont write that yourself
    def __init__(self, name, age):
        '''Creates a new dog object'''
        # Self refers to the current instance
        self.name = name
        self.age = age
        # The convention for marking something as private is to start it with an underscore
        self._tricks_done = 0
        Dog.total_tricks_done += 1

    def sit(self):
        '''Simulate a dog sitting'''
        print(f'{self.name} is now sitting')
        self._tricks_done += 1
        Dog.total_tricks_done += 1

    def roll_over(self):
        '''Simulates a dog rolling over'''
        print(f'{self.name} is now rolling over')
        self._tricks_done += 1
        # Here's a block of code, it does nothing.
        # Use pass in loops, or conditionals for a block of code that does nothing
        # pass

    # Don't need to do this, This is called a Type Hint.
    # def get_tricks_done(self) -> int:
    def get_tricks_done(self):
        '''returns number of tricks done by the dog'''
        return self._tricks_done

    def get_total_tricks_done():
        return Dog.total_tricks_done


fido = Dog('Fido', 2)
print(fido)
print(f'Dogs name is {fido.name}')
fido.sit()
fido.roll_over()
print(f"{fido.name} is {fido.age} years old")
fido.age += 1
print(f"{fido.name} is {fido.age} years old")
print(f'Fido has done {fido.get_tricks_done()} tricks')

sally = Dog('sally', 4)
print(f'fido has done {fido.get_tricks_done()}')
print(f'Sally has done {sally.get_tricks_done()}')
print(f'Dogs in total jabe done {Dog.get_total_tricks_done()}')
sally.roll_over()
sally.sit()
sally.sit()
print(f'fido has done {fido.get_tricks_done()}')
print(f'Sally has done {sally.get_tricks_done()}')
print(f'Dogs in total jabe done {Dog.get_total_tricks_done()}')

# del Dog.get_tricks_done
# print(sally.get_tricks_done())
