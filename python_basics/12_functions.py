# No function hoisting
# say_hello()

# Define a function with def
# Triple quotes s a docstring, that describes the function
def say_hello():
    """Say hello."""
    print('Hello!')


say_hello()

# Can use positional params
def say_hello_name(name):
    """Says hellos to a named person"""
    print(f'hello, {name}')


say_hello_name('john')

# can use more than one param
def say_hello_with_title(title, last_name):
    """Says hellos using a title"""
    print(f'Hello, {title.title()} {last_name.title()}')

say_hello_with_title("mr", 'janzen')

# Can have optional params while including a default
# NOTE: Optional params must come AFTER non optionals
def greet(name, greeting='hello'):
    """Greets someone by name with an option greeting"""
    print(f'{greeting} {name}!')

greet('John', 'Howdy')
greet('John')

# Can have a return type
def get_polite_greeting(name, title, greeting='hello'):
    """Returns a polite greeting"""
    polite_greeting = f'{greeting}, {title} {name}'
    return polite_greeting

my_greeting = get_polite_greeting('janzen', 'Mr', 'Welcome')
print(my_greeting)

# If you want an optional arg, the default can be '' or None
def get_optionally_polite_greeting(name,title='',greeting='hi'):
    '''Generates an optionally polite greeting'''
    if title:
        polite_greeting = f'{greeting}, {title} {name}.'
    else:
        polite_greeting = f'{greeting} {name}!'
    return polite_greeting

my_greeting = get_optionally_polite_greeting('Janzen', 'Mr', "Hello")
print(my_greeting)
my_greeting = get_optionally_polite_greeting('John')
print(my_greeting)

# Can pass in the named arguments if you want
my_greeting = get_optionally_polite_greeting(name='John',greeting='Helloooooo')
print(my_greeting)

def get_greeting_with_bill(name, total=None):
    if total == None:
        greeting = f'Hello, {name}, you owe us {total} dollars'
    else:
        greeting = f'Hello {name}, you dont have an account with us'
    return greeting

print(get_greeting_with_bill('john'))
print(get_greeting_with_bill('wade', 100))
print(get_greeting_with_bill('ron', 0))

# The following values are falsy in Python:
# Number 0
# Boolean value False
# The empty string
# None
# Empty List
# Empty Tuple
# Empty Dictionary

if None == False:
    print("None is equal to False")
else:
    print("None is not equal to false")

if None:
    print("Print None is not falsy")
else:
    print("None is falsy")