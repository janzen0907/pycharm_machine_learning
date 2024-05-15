heroes = ["thor", "ironman", "hulk", 'hawkeye']

for hero in heroes:
    print(f"the next here is: {hero}")

print("Done looping")

# Easy to generate a list of numbers using range, does not include the last index
for number in range (1,5):
    print(number)

# Starts at 0 if just one value is passed in
for number in range (5):
    print(number)

# Third argument that allows you to skip some amount
print("Skipping 2")
for number in range (1, 10, 2):
    print(number)

# No C style loops, can just use a range
for index in range(len(heroes)):
    print(f"Hero # {index} is {heroes[index]}")

# Range returns an object, doesn't return a list.
my_number = range(1,5)
print(my_number)
my_number = list(range(1,5))
print(my_number)

#  List functions
print("Min: ", min(my_number))
print(max(my_number))
print(sum(my_number))

# Assemble a list of the squares of numbers 1 to 10
squares = []
for value in range(1, 11):
    squares.append(value**2)

print(squares)

# Theres a way to do the above more compact
# Called a List Comprehension
cubes = [value**3 for value in range(1,11)]
print(cubes)

# Easy to work with parts of list with slice
planets = ['mecury', 'venus', 'earth', 'jupitur', 'saturn', 'uranus', 'neptune', 'pluto']
print("Planets: ", planets)
# Slice to get specific items like substring
print(planets[3:6])
# Starts at 0 without a starting index
print(planets[:2])
# Omit the last index it will go to the end
print(planets[3:])
# Certain number of elements counting from the end, use a negative
print(planets[-3:]) # Last 3
print(planets[-3:-1]) # Last 2
# With skipping elemetns
print(planets[2::2])
print(planets[::-1]) # Go's backwards

# Loop through a slice
for planet in planets[3:]:
    print(planet)
