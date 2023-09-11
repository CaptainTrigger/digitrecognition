import keras.src.utils
from keras.datasets import mnist
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

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

augmented_images = datagen.flow(train_x, batch_size=1, shuffle=True)

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    random_index = np.random.randint(len(train_x))
    augmented_image = next(augmented_images)[0]
    plt.imshow(augmented_image.reshape(28, 28), cmap='gray')
    plt.title(f"Sample {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
