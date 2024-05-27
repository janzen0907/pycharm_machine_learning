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

# df_crop = pd.read_csv("D:/pycharm_machine_learning/Data/Crop_Recommendation.csv")
df_crop = pd.read_csv("D:/Pycharm/Data/Crop_Recommendation.csv")
print(df_crop.head().to_string())
print(f"Shape is {df_crop.shape}")
# %%


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

# First model, uncomment this to return to old results
model.add(layers.Dense(7, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(len(crop_types), activation='softmax'))
# %%


model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
history = model.fit(x, y, epochs=100, validation_split=.4, batch_size=32)

loss_chart(history)
acc_chart(history)
# %%

def predict_crop(input_data):
    """Function to predict the crop and map the index to a crop name"""
    prediction = model.predict(input_data)

    # Ensure prediction is 2-dimensional
    if prediction.ndim == 1:
        prediction = np.expand_dims(prediction, axis=0)

    # Get indices of the top 2 predictions for each input
    top_2_indices = np.argsort(prediction, axis=1)[:, -2:][:, ::-1]

    # Map indices to crop names
    return [[int_to_crop[idx] for idx in indices] for indices in top_2_indices]

# Test code
# %%
# Expecting Rice
rice_array = np.array([[90, 42, 43, 20.87, 82.00, 6.50, 202.93]])
predicted_crop = predict_crop(rice_array)

# Expecting Maize
maize_array = np.array([[77,57,21,24.93,73.80,6.55,79.74]])
predicted_maize = predict_crop(maize_array)

# Expecting ChickPea
chickpea_array = np.array([[40, 72, 77, 17.02, 16.9, 7.48, 88.55]])
predicted_chickpea = predict_crop(chickpea_array)

# Expecting Kidney Beans
beans_array = np.array([[13, 60, 25, 17.1, 20.59, 5.68, 128.25]])
predicted_beans = predict_crop(beans_array)

print(f"Predicted crop: {predicted_crop}")
print(f"Should be maize: {predicted_maize}")
print(f"Should be chickpea: {predicted_chickpea}")
print(f"Should be Beans: {predicted_beans}")


# %%