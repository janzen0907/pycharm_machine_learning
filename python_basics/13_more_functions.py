
def greet_users(names):
    '''Says hello to a list of users'''
    for name in names:
        print(name)

my_users = ['Alice', 'Bob', 'Candice']
greet_users(my_users)

# Keep in mind a function gets the same reference to a list
def delete_bob(names):
    '''Don't like bob, need to remove him'''
    while 'Bob' in names:
        names.remove('Bob')
    print("The post bob list is: ", names)

# Could pass in a copy to the list so the funciton will not change it
delete_bob(my_users[:])
# Bob is missing from this list of users.
print(my_users)

def change_name(name):
    '''Changes teh name to Bob'''
    name = 'Bob'
    print(f'The name is now {name}')

my_name = 'wade'
print(my_name)
# Would print bob
change_name(my_name)
# Would still be Wade
print(f'my name is {my_name}')

my_users = ['Alice', 'Bob', 'Candice']
func_users = my_users
func_users.remove('Bob')
print(my_users)

my_name = 'wade'
func_name = my_name
func_name = 'Bob'
# THis will be wade
print(my_name)
# This would be bob
print(func_name)

# Pass in a arbritrary number of argumenst, but not in a list
# You can make an parameter a tuple with *
def display_users(*users):
    '''Prints out the users'''
    for user in users:
        print(f'User: {user}')
    print(users)

display_users('Wade', 'bryce', 'ron')

# If you mix positional with arbitrary ones, the arbitrary ones have to come last
# People will call this 'collector' arguments * args
def create_pizza(size, *toppings):
    '''Creates a pizza string'''
    pizza_string = f'{size} pizza with'
    for topping in toppings:
        pizza_string += f'\n{topping}'
    return pizza_string

print(create_pizza('large', 'bacon', 'chicken', 'peppers'))

# Instead of accepting a tuple you can instead set a arbitary set of key value pairs as a dict
# use ** for that
# Often you'll see **kwargs used to collect nonspecific argumes
def create_user_profile(first, last,**user_info):
    '''Creates a user profile'''
    user_info['name'] = first + last
    return user_info

# the arguments require a label to do this
user = create_user_profile('Wade', 'Lahoda', role='Instructor', program='CST')
print(user)

stuff = 'foo'

def do_stuff():
    '''Does stuff'''
    # Accessing a global var
    print(f'stuff is {stuff}')
    # This would cause a error, it thinks that stuff will be a local var rather than a global var
    #stuff = 'bar'
    # Local to th efunction
    my_stuff = 'more stuff'

do_stuff()
stuff = 'bar'
do_stuff()
# print(my_stuff)