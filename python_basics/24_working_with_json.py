from pathlib import Path
import json

path = Path("./my_settings.json")

# settings = {}
# settings['username'] = input('What is your username?')
# content = json.dumps(settings)

# path.write_text(content)

# As  a try block
# try:
#     settings = json.loads(path.read_text())
#     print('We Read:')
#     print(settings)
# except FileNotFoundError:
#     print("No settings yet. Creating them")
#     settings = {}

# Instead of using try we could check if the file exists
if path.exists:
    settings = json.loads(path.read_text())
    print('We Read:')
    print(settings)
else:
    print("No settings yet. Creating them")
    settings = {}

if 'username' in settings:
    print(f"Your current username is {settings['username']}")
else:
    print("You have no username set")

new_username = input("Enter username to save")
if new_username:
    settings['username'] = new_username

if 'language' in settings:
    print(f"You're current default language is {settings['language']}")
else:
    print("YOu have no default language")

new_language = input("Enter deafult language to save")
if new_language:
    settings['language'] = new_language

print(settings)
del settings['password']
path.write_text(json.dumps(settings))