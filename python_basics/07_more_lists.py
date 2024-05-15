# What about Copying a list
colors = ['red','green','blue', 'yellow', 'purple', 'teal']
print(colors)
new_colors = colors
print(new_colors)
new_colors[3] = 'orange'
print(new_colors)
print(colors)

# Not a copy of the list, copy of the reference
# Vars aren't labels for values, they're labels for references that point to objects that contain or represent the value
# Everything is an object in python

# Easiest way to copy the list is to slice the whole list
new_colors = colors[:]
new_colors[5] = 'mauve'
print(colors)
print(new_colors)

# If you want a list that cannot change, that's what tuples are for
instructors = ('wade', 'bryce', 'ernesto','jason')
print(instructors)
print(instructors[2])
# Can have a tuple with only one element, the comma makes it a tuple
tuple_of_one = 'ernesto',
print(tuple_of_one)

# This will generate a TypeError - tuple does not support item assignment
# instructors[1] = 'ron'
# Can create a new value for the tuple, but we cannot change indivdiual items as shown above
instructors = ('wade', 'ron', 'ernesto', 'jason')
print(instructors)

