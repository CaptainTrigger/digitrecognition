import keras.src.utils
from keras.datasets import mnist
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import numpy as np

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = keras.src.utils.normalize(train_x, axis=1)
test_x = keras.src.utils.normalize(test_x, axis=1)

train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)


def add_noise(image, noise_level=0.04):
    noisy_image = image + noise_level * np.random.normal(0, 1, image.shape)
    return np.clip(noisy_image, 0, 1)


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    preprocessing_function=add_noise
)

model = keras.models.Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', min_delta=0.005,
                               restore_best_weights=True)

history = model.fit(datagen.flow(train_x, train_y, batch_size=500),
                    epochs=100,
                    validation_data=(test_x, test_y), callbacks=[early_stopping])

model.save('my_digit_recognition.model')
