import jupyter
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from PIL import Image

# %%
# CSV file built in we can work with
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"X training shapes is: {x_train.shape}\n X test shap is, {x_test.shape}\n Y Training shape is {y_train.shape}")

# %%
plt.imshow(x_train[129])
plt.show()
print(f"image 129: {x_train[129]}")

new_image = Image.fromarray(np.array(x_train[125]))
new_image.save("images/number.png")


# Here we will normalize the values for the images
# %%
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255



# The values are now converted to a value between 0 and 1
print(f"image 129: {x_train[129]}")

# %%
# Add an extra dimension, was a 2D array now 3D
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)
print(f"Train shape is now {x_train.shape}")

input_shape = (28,28,1)
num_classes = 10
print(y_train[0])
# Turns each results into a category
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train[0])

# %%
# Build the model
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32,kernel_size=(3,3), activation='relu'),
    # This will cut down the data set, will take the largest value
    # From a sqaure block of 4 options
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes,activation='softmax')
])

# %%
# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

# %%
# Going to look at the metrics
score = model.evaluate(x_test , y_test, verbose=0)
print(f"Loss evauluations: {score[0]}")
print(f"Accuracy evaluation: {score[1]}")

# Save the model
model.save("Models/numbers.keras")

# %%


