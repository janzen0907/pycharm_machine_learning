import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv("Data/kc_house_data.csv")

# print(df.head().to_string())

# print(f"\n Shape and size \n {df.shape}")
# print(df["price"].describe)
# print(df.dtypes)

# Date is currently a string so we have to change it
# Create new filed called reg_year and take that tp be the first 4 chars of the date
df['reg_year'] = df['date'].str[:4]
df['reg_year'] = df['reg_year'].astype('int')

# print(f"{df.head().to_string()}\n{df.dtypes}")

# Add a new series called house_age to the data frame
# Must conform to the following rules
# If the house was not renovated the age of the house will be the difference betwen the reg_year and the build year
# Else the house age will be the difference between the reg_yera aand the rennovation year

df['house_age'] = np.NAN

# Take 2 vars, i counts from 0, j will be the renovated year value
# Go through every row and again J contains teh value i contains the index
for i,j in enumerate(df['yr_renovated']):
    # Check if it has been renovate, 0 means no
    if j == 0:
        # House age = year registered - year built
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_built']
    else:
        df.loc[i:i, 'house_age'] = df.loc[i:i, 'reg_year'] - df.loc[i:i, 'yr_renovated']

# print(df.head().to_string())

# We want to get rid of the unecessary data fields now
# Axis 0 would identify the row, Axis 1 would identy the column, this is why we are dropping the entire columns
df.drop(['yr_built', 'date', 'yr_renovated', 'reg_year'], axis=1, inplace=True)
df.drop(['id', 'zipcode', 'lat', 'long'], axis=1, inplace=True)
print(df.head().to_string())

# Generally we would have to  check for bad data values
# This would consist of going through each of the series to see if there was bad data
# This ws done beforehand and it was deteremined that there is bad daa with respect to some house age
print("Bad Data\n")
df_bad = df[df['house_age'] < 0]
print(df_bad.to_string())

# Reset the data frame to only contain entries where house age is greater than 0
df = df[df['house_age'] >= 0]
print(f"Good Data \n {df.head().to_string()}")
# print(df.shape)

# for i in df.columns:
#     sb.displot(df[i])
#     plt.show()

# Tons of info returned here,none is readable
#sb.pairplot(df)

# HEat map can be used to show the correaltaion between different series
# The heat map shows corallation, brighter colors is high corralation, dark collors is low corralation

plt.figure(figsize=(30,20))
sb.heatmap(df.corr(), annot=True)
plt.show()


