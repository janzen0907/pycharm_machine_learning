import jupyter
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from PIL import Image
import cv2

# %%
model = keras.models.load_model("Models/numbers.keras")

new_image = Image.open("images/number.png")


plt.imshow(new_image)
plt.show()

# Convert this to a 28 x 28 Image using CV2
# %%
new_image = Image.open("images/seven.jpg")
new_image = cv2.imread("images/seven.jpg", cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(new_image, (28,28))
new_image = cv2.bitwise_not(new_image)
plt.imshow(new_image)
plt.show()


# %%
new_array = np.array(new_image)
new_array = new_array.reshape((1,) + new_array.shape)
print(f"Shape is now: {new_array.shape}")
print(model.predict(new_array))
# print(model.predict(my_four))
# print(model.predict(my_seven))
