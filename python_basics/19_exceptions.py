# Division calculator
print("Give me two numbers, ill divide the first by the second")
print('Enter q to quit')

while True:

    first_number = input("\nFirst Number: ")
    if first_number == 'q':
        break
    second_number = input('\nSecond number: ')
    if second_number == 'q':
        break

    try:
        answer = int(first_number) / int(second_number)
    except ZeroDivisionError as ze:
        print("You cannot divide by zero...")
        continue
    except ValueError as ve:
        print("You need to enter a number")
        continue

    # print(f'{first_number} divided by {second_number} is {answer}')
    # C style print
    print('%s divided by %s is %.2f' % (first_number, second_number, answer))

print('Thank you for using the calculator')