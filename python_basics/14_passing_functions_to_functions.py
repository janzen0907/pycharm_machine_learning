# Everything is an object including functions, can pass functions to other functions

def say_hello(name):
    print(f'hello, {name}!')

def do_something(something):
    something('Something')

# Don't use () if calling a function with another function will cause errors
# do_something(say_hello())

do_something(say_hello)

# Functions are objects - they have a "magic" method called __call__ which
# gets created by the def statement (with other stuff) and gets called
# When you invoke a function
# Dunder function, Double underscore
say_hello.__call__('call')

print(type('hello'))
print(type(say_hello))