languages = ["Python", "C", "C#", "JS"]
print(languages)
# Can sort a list with sort
# This changes the list, if we were to print the list it would actually return None
languages.sort()
print(languages)

# sort can take an optional arg to reverse
# Boolean values True and False are capitalized
# All of Pythons build in constans are PascalCased
# More examples: None - Absecense of a value, NotImplemented, Ellipsis
languages.sort(reverse=True)
print(languages)

# If you want to temp sort you can use sorted
print(languages)
print(sorted(languages)) # Returns a sorted list, does not change the values
print(languages) # Does not change the original list

# Reverse a list - no sorting, just reversing it
years = [2020, 1997, 1999, 1852, 2021]
print(years)
years.reverse()
print(years)

print(len(years))
# Probably shouldn't use this as it is meant to be an internal method
print(years.__len__())

# Access the last item in a list, use -1 as the index
print(years[-1])


