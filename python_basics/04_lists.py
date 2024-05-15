# No arrays. Has lists
# List is a collection of items in a particular order.
# Pythonic code convention for lists is they should be plurals
colors = ['red', "green", "purple"]
print(colors)
# Access an items with []
print(colors[2])

# Can chain with Functions attached to the values
print(colors[2].title())
# Or in a f String
print(f"The third color is: {colors[2]}")
# Can re assign list items
colors[2] = "Yellow"
print(colors[2])
# Generates an error if it doens't exist
# I did a try except just for fun
try:
    print(colors[9])
except IndexError as e:
    print("Out of bounds")

# Add to a list with append
colors.append('teal')
print(colors)
# Lists don't have a fixed size
# Can insert with insert
colors.insert(2, "magenta")
print(colors)

# Remove from a list. Use teh del keyword
del colors[3]
print(colors)

# YOu can use del for other things other than just lists
# my_name = "Wade"
# del my_name
# print(my_name)

# instead of deleting we may want to remove and get the value back, use pop for that
print("Popped value is: ", colors.pop())
# Can supply an index to pop
print(colors.pop(1))
print(colors)

# If you don't the index you can remove using the value
colors.remove('red')
print(colors)
# Multiple instaces of a value, it will remove the first value

