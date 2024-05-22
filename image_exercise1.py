import jupyter
from pathlib import Path
# Pil is the python imaging library, install pillow to get access to it
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, Sequential


# %%

# Exercise 1
# Convert the 3D array into a single value
# Axis 2 is what we are modifying.
# Average each color
#dog_dir = Path("D:\pycharm\Data\cat_dog_images")
animal_dir = Path("D:\pycharm\Data\cat_dog_images")
# print(dog_dir.absolute())

# all_files = list(dog_dir.glob("*.png"))
animal_files = list(animal_dir.glob("./*/*.png"))

for animal_file in animal_files:
    animal_name = animal_file.name
    parent_dir = animal_file.parent.name

    print(f"{animal_name}'s parent dir is {parent_dir}")
    animal_image = Image.open(str(animal_file))

    animal_array = np.array(animal_image)
    # Specifying the axis here allows us to average the color values
    bw_animal_array = np.average(animal_array, axis=2)
    plt.imshow(Image.fromarray(bw_animal_array))
    plt.show()

    bw_animal_array = bw_animal_array.astype(np.uint8)

    # Let's make sure the parent dir exists
    Path(f"D:\pycharm\Data\monochrome\{parent_dir}").mkdir(exist_ok=True, parents=True)
    bw_file_path = Path(f"D:\pycharm\Data\monochrome\{parent_dir}/{animal_name}")

    bw_animal_image = Image.fromarray(bw_animal_array)
    bw_animal_image.save(bw_file_path, "PNG")

    plt.imshow(bw_animal_image)
    plt.show()

# %%