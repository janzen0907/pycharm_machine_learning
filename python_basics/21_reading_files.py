# Python provides the pathllb modules that makes It easier to work with files and directories
# without needing to worry about the OS details
from pathlib import Path
# Get current path
print(Path().absolute())

# Create a path object pointing to a file we want to read.
path = Path("./requirements.txt")
# Note: Even in Windows, using forward slashes when using pathlib

# There will be a blank line at the end becasue the library will return an empty string
# when it reaches the end f the file. YOu can strip that out if you want.
contents = path.read_text().rstrip()
print(contents)

# May want to deal with stuff line by line
lines = contents.splitlines()
lineno = 1
for line in lines:
    print(f"{lineno}: {line}")
    lineno += 1

# Can use various string methods on the contents. Can use
# sequence methods like slicing [2:6].

# We want requirements.txt with just the package names, no version numbers, all on one line, comma seperated
requirements_string = ''
for line in lines:
    if requirements_string:
        requirements_string += ', '
    requirements_string += line[:line.find('==')]

print(requirements_string)



