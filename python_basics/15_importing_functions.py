import pythonmodule

new_user = pythonmodule.create_user_profile("Wade", "Lahode", role='instructor')
print(new_user)

# often will import with an alias
import pythonmodule as pm
pm.greet_users(['Alice', 'bob'])

# Import only one function
from pythonmodule import create_pizza
print(create_pizza('large', 'pepperoni'))

from pythonmodule import display_users as du
du('wade', 'bryce', 'ron')

# Something you can do but shoudlnt, import all wiht *
from pythonmodule import *