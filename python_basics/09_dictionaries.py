# Use dictionaries alot
# List of key value pairs
# Key colon value comma
# Reference components by using square brackets and the key
pepperoni_pizza = {'name': 'pepperoni', 'price': 12.50, 'sauce':'red'}
print(pepperoni_pizza)
print(pepperoni_pizza['price'])

# ANything can be the value, and anything can be the key if it is hashable
# Should keep to integers and strings for keys
movies = {
    54: 'E.T',
    72: 'Short Circuit',
    65: 'The last Starfighter'
}
print(movies[72])

# Can add keys dynmaically
pepperoni_pizza['topping'] = 'cheese'
print(pepperoni_pizza)

canadian_pizza = {}
canadian_pizza['name'] = 'canadian'
canadian_pizza['price'] = 14.50
canadian_pizza['sauce'] = 'red'
print(canadian_pizza)

for key in pepperoni_pizza:
    print(f"Key {key} is {pepperoni_pizza[key]}")

# YOou can delete keys with del
del pepperoni_pizza['price']
for key in pepperoni_pizza:
    print(f"Key {key} is {pepperoni_pizza[key]}")

# Try to access a key that does not exist
# Generates a key error
try:
    print(pepperoni_pizza['size'])
except KeyError as ke:
    print("The key does not exist")

# If worried about a key not existing you can use get
# Returns None if it does not exist
pizza_size = canadian_pizza.get('size')
print(pizza_size)
if pizza_size:
    print(f"Our pizze is {pizza_size}")
else:
    print("We dont know the size")

pizza_size = canadian_pizza.get('sauce')

# Go through keys and values in the list
print(pepperoni_pizza.items())
for key, value in pepperoni_pizza.items():
    print(f"key is {key}, and value is {value}")