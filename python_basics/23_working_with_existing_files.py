from pathlib import Path

# Read an exisiting file suggested by the user

filename =input("What file would you like to add some content to?")
path = Path(filename)

# Fine to open a file taht doensn't exist if all you are doing is writing.
# But what if you want to read?

# read in the exisiting contents of the file
try:
    contents = path.read_text()
except FileNotFoundError as e:
    print(f"File does not exist, we will create it")
    contents = ''

contents += '\n' + input("Enter the text you want to add to the file\n")
print("Writign:")
print(contents)
path.write_text(contents)


