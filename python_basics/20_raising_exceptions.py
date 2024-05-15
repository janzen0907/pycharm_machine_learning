def divide(first,second):
    return (int(first) / int(second))

divide(5, 10)
try:
    divide(10, 0)
except ZeroDivisionError:
    print("You cant divide by 0")
    # Sometimes you will want to re-raise an exception to pass it on up the chain
    # raise

# Not all exceptions are errors - they're just exceptions to the normal flow
# StopIteration, for instance, is an exception used to terminate iteration over iterators
# If an exception is an Error, it'll end in Error
# Theres also exceptions that end in warning
# raise Exception("There was a problem", 10, 5)

