import jupyter
import keras
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras import layers, models


# %%
#
def acc_chart(results):
    plt.title("Accuracy of Model")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# Function from class to display loss chart
def loss_chart(results):
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# Function from class to show heatmap
def do_visual(df_sample):
    sb.heatmap(df_sample.corr(), annot=True)
    plt.show()


# Read in the CSV file
# print(loan_info.head().to_string())
# df_loan = pd.read_csv("D:/pycharm_machine_learning/Data/loan.csv")
df_loan = pd.read_csv("D:\pycharm\Data\loan.csv")

df_loan = df_loan.drop("occupation", axis=1)

# print(df_loan.head())
# Change the strings to int values
df_loan['education_level'] = df_loan['education_level'].map({"High School": 0, "Bachelor's": 1, "Master's": 2})
df_loan['gender'] = df_loan['gender'].map({"Male": 0, "Female": 1})
df_loan['marital_status'] = df_loan['marital_status'].map({"Married": 0, "Single": 1})
df_loan['loan_status'] = df_loan['loan_status'].map({"Approved": 0, "Denied": 1})


def show_histograms(df):
    """Function to display the data in histograms"""
    df_approved = df['loan_status'] == 0
    df_denied = df['loan_status'] == 1

    # Compare loan status against age
    plt.hist(df[df_approved]['age'], color='b', alpha=0.5, bins=15, label="Approved")
    plt.hist(df[df_denied]['age'], color='g', alpha=0.5, bins=15, label="Denied")
    plt.legend()
    plt.title("Loan Status vs Age")
    plt.show()

    print(df.head())
    # Compare loan status against education
    plt.hist(df[df_approved]['education_level'], color='b', alpha=0.5, bins=5, label="Approved")
    plt.hist(df[df_denied]['education_level'], color='g', alpha=0.5, bins=5, label="Denied")
    plt.legend()

    # Set custom x-ticks to show strings instead of the floats
    plt.xticks(ticks=[0, 1, 2], labels=["High School", "Bachelor's", "Master's"])
    plt.title("Loan Status vs Education")
    plt.show()

    # Compare loan status against marital status
    plt.hist(df[df_approved]['marital_status'], color='b', alpha=0.5, bins=5, label="Approved")
    plt.hist(df[df_denied]['marital_status'], color='g', alpha=0.5, bins=5, label="Denied")
    plt.legend()
    plt.xticks(ticks=[0, 1], labels=["Married", "Single",])
    plt.title("Loan Status vs martial status")
    plt.show()


x = df_loan.drop("loan_status", axis=1)
y = df_loan['loan_status']

# print("Shape of x is %s " % str(x.shape))
# print("Shape of y is %s " % str(y.shape))

show_histograms(df_loan)

model = models.Sequential()
# %%

# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(48, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(4, activation='relu'))
# model.add(layers.Dense(2, activation='relu'))
# model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history = model.fit(x, y, epochs=150, validation_split=0.2, batch_size=64)
loss_chart(history)
acc_chart(history)

# %%
# Save the mode
model.save("D:/pycharm/Models/loan.keras")
