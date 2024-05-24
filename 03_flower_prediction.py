import PIL
import jupyter
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, Sequential

# Define some consts
IMG_HEIGHT=180
IMG_WIDTH=180
CLASSES = ['daisy','dandelion','roses','sunflowers','tulips']

model_path = Path('./data/models/flower_models.keras')
flower_model = keras.models.load_model(model_path)
print(flower_model.summary())

while True:
    flower_file = input("Enter the path of an image you'd like ot be identified")
    flower_file_path = Path(flower_file)
    if not flower_file_path.is_file():
        print(f"{flower_file} is not a file. Please enter the path to an image ")
        continue


    # Use keras to load in th efile
    try:
        img = keras.utils.load_img(flower_file_path, target_size=(IMG_HEIGHT,IMG_WIDTH))
    except PIL.UnidentifiedImageError:
        print(f"{flower_file} does not appear to be a valid image")
        continue

    print("Predicting..")
    # We want to transform the image into a numpy array
    img_array = np.array(img)
    # Keras expects us to be passing in an array of images not a singel image
    # So we need to wrap it in the 4th dimension
    img_array = img_array.reshape((1,) + (IMG_HEIGHT,IMG_WIDTH,3))


    # Pass our size 1 array of images inot our model and see what it predicts.
    predictions = flower_model.predict(img_array)
    print(predictions)
    # Let's transform it into a "score"
    score = tf.nn.softmax(predictions[0]) * 100
    print(score)
    predicted_class = CLASSES[np.argmax(score)]
    confidence = round(np.max(score),2)
    print(f"This image is of class {predicted_class} with a {confidence} % confidence")

# Paths for images
# D:\random flowers\daisy1.jpg
# D:\random flowers\daisy2.jpg
# D:\random flowers\flower1.jpg
# D:\random flowers\flower2.jpg
# D:\random flowers\tulips1.jpg
# D:\random flowers\tulips2.jpg

# %%