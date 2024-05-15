print(2 + 3)
print(2 - 3)
print(2*3)
print(2/3)
# Use ** for exponents
print(2**3)

# Follows order of operation BEDMAS
print(2 + 3 -4 +5 /6 ** 7)


# Can mix intergers and floats together, you'll just get a float when you do
my_int = 2
my_float = 3.0
print(my_int + my_float)

# Conveince feature for large numbers, is you can put underscores in numbers and python ignores then
big_number = 99_234_865_321
print(big_number)

# Python is strongly, dynamically typed
# Strongly in the sense taht the type of a value (not a variable but the value inside) doesn't change without explict
# conversion dynamically in that the runtime objects - values - have a type, but a variable does not
# You can put whatever you want inside a variable and teh interpeter will not care
# BUT, you can't for instance, treat a string as a number without explicitly saying so
my_string = "13"
my_number =  int(my_string) * 3
print(my_number)