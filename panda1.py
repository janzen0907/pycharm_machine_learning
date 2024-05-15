import numpy as np
import pandas as pd

# create a panda series
series1 = pd.Series([12,14,17,19,20], index=["one", "two", "three", "four", "five"])
print(series1)

series2 = pd.Series([33,22,11,10])
print(series2)

# Create a series from a dict
workout = {"Mon": "Legs", "Tue": "Core", "Wed": "Biceps", "Thurs": "Rest", "Fri": "Leg"}
sWorkout = pd.Series(workout)
print(sWorkout)
print(sWorkout.loc["Mon"])

print(series2.loc[1])

df2 = pd.DataFrame(
    {
        "IDS": pd.Series([10, 11, 12, 13]),
        "Names": pd.Series(["Paul", "George", "Ringo", "John"]),
        "Status": pd.Categorical(["Alive", "Dead", "Alive", "Dead"]),
        "Band": "Beatles"
    }
)

# Add another column of data to the data frame
plays = ["Sings", "Bass", "Drums", "Sings"]
df2['plays'] = plays

# Add a new row
# 1 We can get the length (# rows) of the dataframe by using the len(index property)
# We can then add at that location by just creating the appropriate row entry
df2.loc[len(df2.index)] = [14, "Mike", "Alive", "Beatles", "Drums"]



print(df2)
print(df2.dtypes)

# Can access indivdual rows by using loc value
# Print all rows in table
print(df2.loc[::])
# 2 to the end
print(df2.loc[2:])

# print out a single row
print("\n")
print(df2.loc[1:1])

# Print out only the column entries for the Names
print("\n")
print(df2.loc[::, "Names"])

print("\n")
print(df2.loc[::, ["Names", "Status"]])
# Specify row first then to column

# We can access only the values we are interested in
# For the following we will list the entries assocatied with the dead beatles
dfDead = df2[df2["Status"] == "Dead"]
print(f"\nDead Guys:\n{dfDead}\n")

dfAlive = df2[df2["Status"] == "Alive"]
print(f"\nAlive Guys:\n{dfAlive}")

# Bryce was having issues here
# dfIds = df2[df2["Status"] == "Dead"["IDS"]]

