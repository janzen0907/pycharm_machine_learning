import keras
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras import layers, models

my_model = keras.models.load_model("Models/wine.keras")

md_wine = np.array([[7.5,0.71,0,1.8,0.76,10.88,34.2,0.00,3.5,0.55,9.5]], dtype=np.float64)
y_predict = my_model.predict(md_wine)
print(y_predict)