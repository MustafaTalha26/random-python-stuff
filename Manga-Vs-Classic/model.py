import keras
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
from keras.applications import MobileNetV2
warnings.filterwarnings('ignore')

trainAug = keras.models.Sequential([
	keras.layers.Rescaling(scale=1.0 / 255),
	keras.layers.RandomFlip("horizontal_and_vertical"),
	keras.layers.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	keras.layers.RandomRotation(0.3)
])

def MobileNetV2_model(input_shape):
    baseModel = MobileNetV2(include_top=False, input_tensor=keras.layers.Input(shape=input_shape))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(trainAug)
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

train = keras.utils.image_dataset_from_directory(
    "Comics/train/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    label_mode='binary',
    shuffle=True,
    interpolation='nearest',
)
valid = keras.utils.image_dataset_from_directory(
    "Comics/validation/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    label_mode='binary',
    shuffle=True,
    interpolation='nearest'
)

#aug_model = data_augmentation_model((224,224,3))

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
size = (224, 224)
shape = (224,224, 3) 
epochs = 10

model = MobileNetV2_model(shape)

model.compile(
    optimizer=keras.optimizers.Adam(2e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train,
    epochs=epochs,
    validation_data=valid,
    callbacks=[callback]
)

