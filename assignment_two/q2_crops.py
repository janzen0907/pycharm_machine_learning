import jupyter
import keras
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras import layers, models

# %%
# Function from class to display accuracy chart
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

df_crop = pd.read_csv("D:/pycharm_machine_learning/Data/Crop_Recommendation.csv")
print(df_crop.head().to_string())
print(f"Shape is {df_crop.shape}")
# %%
# Using this Data Set– create a model that given values for
# the first 7 fields, will identify the top 2 Cops that
# should be planted given the specified conditions.
# Using this Data Set– create a model that given
# values for the first 7 fields, will identify the top
# 2 Cops that should be planted given the specified

# I was getting errors as we cannot cast the string to floats
# So i am going to enumarate all the crops, and assign
# Then numeric values

crop_types = df_crop['Crop'].unique()
crop_to_int = {crop: idx for idx, crop in enumerate(crop_types)}
int_to_crop = {idx: crop for crop, idx in crop_to_int.items()}

df_crop['Crop'] = df_crop['Crop'].map(crop_to_int)

x = df_crop.drop("Crop", axis=1)
y = df_crop['Crop']

model = models.Sequential()
#model.add(layers.Dense(7, activation='relu'))
#model.add(layers.Dense(3, activation='relu'))
#model.add(layers.Dense(6, activation='softmax'))

model.add(layers.Dense(7, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation='relu'))
#model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(len(crop_types), activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=100, validation_split=.2, batch_size=32)

loss_chart(history)
acc_chart(history)


def predict_crop(input_data):
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)
    return [int_to_crop[idx] for idx in predicted_class]


sample_input = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])
predicted_crop = predict_crop(sample_input)
print(f"Predicted crop: {predicted_crop}")
