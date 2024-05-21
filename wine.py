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


def do_visual(df_sample):
    sb.heatmap(df_sample.corr(), annot=True)
    plt.show()

df_wine = pd.read_csv("Data/winequality-red.csv")

print(df_wine.head(4).to_string())
print(f"Shape is {df_wine.shape}")

do_visual(df_wine)

x = df_wine.drop("quality", axis=1)
y = df_wine["quality"]

model = models.Sequential()
model.add(layers.Dense(11, activation='relu'))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Use this loss method when more than 1 classification
model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x,y, epochs=150, validation_split=.2, batch_size=100)

loss_chart(history)
acc_chart(history)

md_wine = np.array([[7.5,0.71,0,1.8,0.76,10.88,34.2,0.00,3.5,0.55,9.5]], dtype=np.float64)
y_predict = model.predict(md_wine)
print(y_predict)

# Save the model to a file which can then be read
model.save("Models/wine.keras")



