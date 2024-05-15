topping = 'mushrooms'
if topping == 'mushrooms':
    print("Gimme mushrooms")

if topping != 'cheese':
    print("I'm lactose")

age = 20
if age >= 19:
    print("can drink in sask")

if age >= 19 and age <=21:
    print("Drink  in sask not in Freedom land")

color = 'red'
if color == 'red' or color == 'white':
    print("color is on our flag")

if not color == 'blue':
    print("Not blue")

# can use in to check if the value is in the list
toppings = ['mushrooms', 'onions', 'pinapple']

if 'mushrooms' in toppings:
    print("You ordered mushrooms")

if 'pepperoni' in toppings:
    print("Peps")

if 'peppers' not in toppings:
    print("Hold the peppers")

# Empty list is falsy
toppings = []
if toppings:
    print("You've asked for some toppings")
else:
    print("You haven't ordered any topings")

# Treated as false but not the same as being false. Can use it in a conditional still
print([]==False)

# No switch statements naturally, theirs a pattern matching construct in 3.10 and later,
# most people use if/else if /else blocks instead

age = 25
if age < 13:
    print("Young Child")
elif age < 18:
    print("Teenager")
elif age < 65:
    print("Adult")
else:
    print('senior')