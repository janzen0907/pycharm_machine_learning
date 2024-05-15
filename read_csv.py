import pandas as pd

# Read in the CSV file
my_planes = pd.read_csv("Data/wwIIAircraft.csv")
#print(my_planes)

# Examples of printing and different functions
# print(my_planes.head(5))
# print(my_planes.tail(5))

# Print out all the planes whose Country of Origin is the us
us_planes = my_planes[my_planes["Country of Origin"] == "US"]
print(us_planes)
