import keras
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras import layers, models


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


def graph_stuff(df_copy):
    # Want to look at the numbers for the participants
    # Changing the type from what it is in the data to words for readability
    df_copy['condition'] = df_copy['condition'].map({0:"Healthy", 1:"Heart Patient"})
    # Changing sex from 0 and 1 to words
    df_copy['sex'] = df_copy['sex'].map({1:"Male", 0:"Female"})

    # Loot at the male graphs fr this
    df_male = df_copy[df_copy['sex'] == "Male"]
    # X axis is condition
    sb.countplot(x='condition', data=df_male)
    plt.title("Male Stats")
    plt.show()

    # Same thing for females
    df_female = df_copy[df_copy['sex'] == "Female"]
    sb.countplot(x='condition', data=df_female)
    plt.title("Female Stats")
    plt.show()

    cond_health = df_copy['condition'] == 'Healthy'
    cond_sick = df_copy['condition'] == 'Heart Patient'

    # Compare age with health
    # Put values in 15 different bins. Alpha is opacity. Color blue.
    plt.hist(df_copy[cond_health]['age'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(df_copy[cond_sick]['age'], color='g', alpha=0.5, bins=15, label="Heart problems")
    plt.legend()
    plt.title("Health count VS Age")
    plt.show()

    # compare chol with health
    plt.hist(df_copy[cond_health]['chol'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(df_copy[cond_sick]['chol'], color='g', alpha=0.5, bins=15, label="Heart Problems")
    plt.legend()
    plt.title("Health Count VS Chol")
    plt.show()

    # compare thalach (max achieved heart rate) with health
    plt.hist(df_copy[cond_health]['thalach'], color='b', alpha=0.5, bins=15, label="Healthy")
    plt.hist(df_copy[cond_sick]['thalach'], color='g', alpha=0.5, bins=15, label="Heart Problems")
    plt.legend()
    plt.title("Health Count VS Thalach")
    plt.show()



df_heart = pd.read_csv("Data/heart_cleveland_upload.csv")

print(df_heart.head().to_string())
print(df_heart.describe())

# Graph some of the fields
# graph_stuff(df_heart.copy())

# Find the outliers, high cholesteral and get rid of it
df_high_col = df_heart[df_heart['chol'] > 500]
print(df_high_col.to_string())
df_heart = df_heart[df_heart['chol'] < 500]

# Low thalach, get rid of it
df_low_thalach = df_heart[df_heart['thalach'] < 80]
print(df_low_thalach.to_string())
df_heart = df_heart[df_heart['thalach'] > 80]

x = df_heart.drop("condition", axis=1)
y = df_heart['condition']

print("Shape of x is %s " % str(x.shape))
print("Shape of y is %s " % str(y.shape))

model = models.Sequential()

model.add(layers.Dense(13, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Loss function here is used just for True/False stats
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training on 80% of the data, testing on 20. 200 times through the data
history = model.fit(x,y, validation_split=0.2, epochs=200)
# If we don't specify a batch size, the default is 32
# We have 295 entires, this means we will run
# 10 different samples of size 32 for each epoch
acc_chart(history)
loss_chart(history)

x_at_risk = np.array([[62,1,3,145,250,1,2,120,0,1.4,1,1,0]], dtype=np.float64)
y_at_risk = (model.predict(x_at_risk) > 0.5).astype(int)
print(y_at_risk[0])

x_health = np.array([[50,1,2,129,196,0,0,163,0,0,0,0,0]], dtype=np.float64)
y_healthy = (model.predict(x_health) > 0.5).astype(int)
print(y_healthy[0])


