
message = "hello python world"
print(message)

message = "hello again"
print(message)

# Variables can only contain letters, numbers and underscores. Cannot start with a number
# Use Snake case

# This will generate an error, interpreter will try to guess what you mean
# print(massage)

# Anything you put inside quotes - wheter single or double - is a string
name = "John Janzen"
# strings have build in method
print(name.title())
print(name.upper())
print(name.lower())

# You can use varibles inside of a string by using f-string, f is for format
last = "janzen"
first = "john"
full_name = f"{first} {last}"
print(full_name)
print(f"hello, {full_name}")
print(first + last)

# Just like you'd expect you can use new lines and tabs
long_message = "Hello, welcome to \t\tIntro to python\n How do you like it?\n"
print(long_message)

# When parsning stuff we want to clean up whitespace. Strings have an rstrip method that will
# return the string with extra spaces on the right removed
extra_spaces = "       This string            "
print(f"The strign rstripped is '{extra_spaces.rstrip()}'")
# Note this does not modify the original string
print("The original string is", extra_spaces)
# Theres an L strip as well
print(f"Lstripped it is '{extra_spaces.lstrip()}'")
# You can chain them together
print(f"lstripped and then rstripped: '{extra_spaces.lstrip().rstrip()}'")

# Remove a prefix with removeprefix
my_url = "http://johnlearnspython.com"
print(f"{my_url.removeprefix('http://')}")

# Concate strings with a plus
first = "First"
second = "second"
print(first + second)
both = first + second
both += "Third"
print(both)

# Me messing around in class
# for i in message:
#     print(i)

# first_word = message[0:5]
# print(first_word)