prompt = "Tell me stuff, ill repeat it"
prompt += "\n(type quit to quit)"

message = ""
# while message != 'quit':
#     message = input(prompt)
#     print(f"You said {message}")

# Re-write this using break
# while True:
#     message = input(prompt)
#     if message == 'quit':
#         break
#     print(f"You said {message}")

current_num = 0
while current_num < 10:
    current_num += 1
    if current_num % 2 == 0:
        continue
    print(f"{current_num} is odd")

# Most of the time will use for in loops with lists
# But shouldn't modify a list while you are looping through it
# If your modifying a list, use a while loop
potential_clients = ['alice', 'bob', 'candice', 'donald', 'edgar']
confirmed_clients = []

while potential_clients:
    current_client = potential_clients.pop()
    accept = input(f"Should we take {current_client} on as a client (type y to accept)")
    if accept == 'y':
        confirmed_clients.append(current_client)

print("confirmed clients are")
for client in confirmed_clients:
    print(client.title())

animals = ['dog', 'cat', 'parrot', 'cat', 'cat', 'dog', 'hamster', 'fish', 'cat', 'parrot']
#  Remove cats
while 'cat' in animals:
    animals.remove('cat')
print(animals)
