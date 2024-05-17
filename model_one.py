import keras
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras import layers

def acc_chart(results):
    plt.title("Accuracy of Model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

def loss_chart(results):
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()



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

# plt.figure(figsize=(30,20))
# sb.heatmap(df.corr(), annot=True)
# plt.show()

# We want to be able to predict the price of a house based upon
# other series values. So we are going to create a DataFrame for
# our input values and another dataframe four output values,
# Both of these become part of our model analysis later on.

x = df.drop("price", axis=1)
y = df["price"]

print(f"{x.head().to_string()}\n {y.head().to_string()}")

print(f" {x.shape} and {y.shape}")

# Define our model
# One input layer, one hidden layer and one output layer
my_model = keras.Sequential()
# This matches up to the number of columns
my_model.add(layers.Dense(14, activation="relu"))
my_model.add(layers.Dense(4, activation="relu"))
my_model.add(layers.Dense(2, activation="relu"))
my_model.add(layers.Dense(1))

# Compile our model and put these into a result series
my_model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
# Batch size - Number of seperate entries
# Epochs number of times throught he system
# Can play with the batch size and epochs. more epochs is generally more accurate
results = my_model.fit(x ,y, validation_split=0.33, batch_size=64, epochs=33)

# Steps to do things in the console
# Copy and paste all the code in the python console
# print an entry
# print(df.loc[781:781].to_string())
# create a sample house with the data from the above print
# samp_house = np.array([[3,1.75,1590,8219,1.5,0,0,5,6,970,620,2030,7504,76]], dtype=np.float64)
# Create the predicted entry with the model
# pred_house = my_model.predict(samp_house)
# print that out
# print(pred_house[0])


