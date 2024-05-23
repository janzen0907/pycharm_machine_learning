import jupyter
from pathlib import Path
# Pil is the python imaging library, install pillow to get access to it
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, Sequential


# %%
print("Hello")

print("Goodbye")
# %%
cat_dir = Path("D:\pycharm\Data\cat_dog_images\cat")
print(cat_dir.absolute())


cat_files = list(cat_dir.glob("*.png"))
print(f"There are {len(cat_files)} cat pictures:")
print(cat_files)

cat_img = Image.open(str(cat_files[0]))

print(f"Our first cat image is {cat_img.size}")
print(f"It's mode is {cat_img.mode}, and its format is {cat_img.format}")

plt.imshow(cat_img)
plt.show()

# %%

cat_array = np.array(cat_img)
print(cat_array)
print(f"The shape of the cat array is {cat_array.shape}.")
# %%

# Since it is an array, we can do whatever aray operations on it we want
new_cat_array = np.where(cat_array < 100, 255, cat_array)
# new_cat_array = cat_array[:,:,:]
# new_cat_array[:,:,0] = 0
# new_cat_array[:,:,1] = 0

new_cat_image = Image.fromarray(new_cat_array)
plt.imshow(new_cat_image)
plt.show()

print(keras.backend.image_data_format())

rolled_cat_array = np.rollaxis(new_cat_array, 2, 0)
print(f"The rolled cat array is shape is: {rolled_cat_array.shape}")
# %%

mutated_cat_array = new_cat_array * 0.33
mutated_cat_array = mutated_cat_array.astype(np.uint8)
mutated_cat_image = Image.fromarray(mutated_cat_array)
plt.imshow(mutated_cat_image)
plt.show()

# %%
# How would we go and save this mutated image?
mutated_cat_file = Path("D:\pycharm\Data\mutated\cat1.png")
# Checking parent dir locations
# print(mutated_cat_file.parents[0])
# print(mutated_cat_file.parent)
# Make the parent dir if it doesn't exist
mutated_cat_file.parent.mkdir(exist_ok=True)
mutated_cat_image.save(mutated_cat_file, "PNG")

# %%
# Let's trying using some Keras layers to augment our images automatically
# Keras will expect an array of images, we're only working with one image,
# so we will have to reshape it from (200,240,3) shape it to (1,200,240,3)
cat_images_array = cat_array.reshape((1,) + cat_array.shape)
print(f"cat_image_arrays shape is {cat_images_array.shape}")

# Lets add some keras preprocessing layers:
image_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(cat_array.shape)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)

# Let's generate a few random variations:
for i in range(10):
    print(f"Image #{i}:")
    augmented_images = image_augmentation(cat_images_array)
    # There's only one image in our array, so we will grab that:
    augmented_image = augmented_images[0]
    plt.imshow(augmented_image.numpy().astype(np.uint8))
    plt.show()
#%%





