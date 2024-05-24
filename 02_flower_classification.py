import jupyter
from pathlib import Path
# Pil is the python imaging library, install pillow to get access to it
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, Sequential

# %%
#config
data_dir = Path("./data/flower_images")
print(data_dir.absolute())


# %%
# Lets Take a loot at the data, check how many images
image_count = len(list(data_dir.glob("*/*.jpg")))
print(f"We have {image_count} .jpg files in our data folder")

# Get a single image
roses = list(data_dir.glob("roses/*"))
for i in range(5):
    print(f"Image number: #{i}")
    my_image = Image.open(str(roses[i]))
    plt.imshow(my_image)
    plt.show()
#%%
# There's a lot of image in here. Lets use a keras utility to load it in as a dataset
# It'l; take in the whole directory just a couple of lines of code
# WE could do this from scrath, using tensorflows data module, if we wanted finer grain control

# Let's define some parameters:
BATCH_SIZE = 32
# Note that our sample images are not uniform in dimension. (Different PX count)
# Lets load them all in as the same size
IMG_HEIGHT = 180
IMG_WIDTH = 180

# We're going to split th eadata into validation data and training data. We'll use 20% for validation
VALIDATION_SPLIT = 0.2

# Create our data sets
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE

)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# image dataset from dir will automaticcaly use the directories as "class names" which
# is generally what we want
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"We have {num_classes} classes")
print(class_names)

#%%
# Let's visualize some of the data loaded in, by dislaying the first 9 images
# from the training dataset
plt.figure(figsize=(10,10))
# We'll take 1 batch of 32 images from our dataset:
for images, labels in train_ds.take(1):
    print("One image batch:")
    print(images.shape)
    print("One label batch:")
    print(labels.shape)

    # This will create the lables and a grid type layout to show all the images
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

# %%
# When we're loading in our data we'll want to configure for performance at least some.
# Two important setting for performance
# - We'll want to cache data in our dataset to keep some images in memory after they're loaded from disk
# - We'll want ot make sure we're prefetching so we overlap data preprocessing and model execution while training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# We're also going to need to standardize the array. RGB channel_values run from 0 -255
# This isn't ideal for a neural network! Ideally, we'd have values from 0 to 1. We can use a
# rescalling layer to normalize our values to be in the 0 to 1 range.
normalizaion_layer = layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH,3))

# We could apply this layer right now using Dataset.map, but we'll just throw it in at the
# start of our model instead.
#%%
# Our next step is to build the model
model = Sequential([
    normalizaion_layer,
    layers.Conv2D(16,3,padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)

])

# We can now compile our modely
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

# %%
# Let's train the model:

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#%%
#Let's visualize the training results and create a plot
# of the loss and accuracy on both th etraining and validation sets:

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.legend(loc='upper left')
plt.title('Training and validation loss')

plt.show()


