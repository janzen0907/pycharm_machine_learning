# Use the input function
message = input("What do you want to print?")
print(message)

# Build up a multi-line promt string first
prompt = "Give us your first name, we'll be able to customize output."
prompt += "\nWhat is your first name?\n"
name = input(prompt)
print(f"Hello, {name}")

# If you enter a number, input will still return a string
birth_year = input("What year were you born?")
age = 2024 - int(birth_year)
print(f"Assuimg {name} was born Jan 1st, {name} is {age} years old")

#float function
string_contains_float = "           035.460        "
print(float(string_contains_float))

fav_num = int(input("Whats your fav number"))
if fav_num & 2 == 0:
    print("Your fav num is even")
else:
    print("Fav numebr is odd")